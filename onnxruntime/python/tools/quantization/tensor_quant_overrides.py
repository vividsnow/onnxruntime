# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

import onnx

from .quant_utils import QuantType


class TensorQuantOverridesHelper(MutableMapping):
    def __init__(
        self,
        raw_overrides: dict[str, list[dict]],
        default_activation_qtype: QuantType,
        default_weight_qtype: QuantType,
    ):
        self.overrides = raw_overrides
        self.default_activation_qtype = default_activation_qtype
        self.default_weight_qtype = default_weight_qtype

    def get_node_output_qtype(self, output_name: str):
        if output_name not in self.overrides:
            return self.default_activation_qtype

        # Get the first overrides dict in the list. This works for both per-tensor and per-channel
        # quantization because all channels must use the same quant type.
        tensor_overrides = self.overrides[output_name][0]

        return tensor_overrides.get("quant_type", self.default_activation_qtype)

    def get_node_input_qtype(self, input_name: str, node_name: str):
        if input_name not in self.overrides or not self.overrides[input_name]:
            return self.default_activation_qtype

        # Get the first overrides dict in the list. This works for both per-tensor and per-channel
        # quantization because all channels must use the same quant type.
        tensor_overrides = self.overrides[input_name][0]
        producer_type = tensor_overrides.get("quant_type", self.default_activation_qtype)

        if "convert" not in tensor_overrides:
            return producer_type

        convert_dict = tensor_overrides["convert"]

        if "recv_nodes" not in convert_dict:
            return convert_dict["quant_type"]  # All consumers converted to the quant_type

        # Only specific consumers get the converted quant_type
        return convert_dict["quant_type"] if node_name in tensor_overrides["convert"]["recv_nodes"] else producer_type

    def update_tensor_overrides(
        self,
        tensor_name: str,
        channels: list[int] | None = None,
        overwrite: bool = True,
        **kwargs,
    ) -> bool:
        if not kwargs:
            return False

        channels = set(channels) if channels is not None else None
        have_overrides = self.overrides.get(tensor_name)

        # If `overwrite` is False, check if we would overwrite anything.
        do_update = True
        if not overwrite and have_overrides:
            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                if set(kwargs).intersection(set(overrides)):
                    do_update = False
                    break

        # Do the update if `overwrite` is True or if nothing is overwritten (do not want partial overwrites).
        if do_update:
            if not have_overrides:
                self.overrides[tensor_name] = [{}]

            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                overrides.update(kwargs)

        return do_update

    def update_converted_tensor(
        self, tensor_name: str, producer_type: QuantType, consumer_type: QuantType, consumer_names: set[str]
    ):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            self.overrides[tensor_name] = [{"quant_type": producer_type}]

        overrides = self.overrides[tensor_name][0]

        if "convert" not in overrides:
            overrides["convert"] = {"quant_type": consumer_type}

        convert_dict = overrides["convert"]
        if consumer_type != convert_dict["quant_type"]:
            raise ValueError(f"Desired consumer quant_type for {tensor_name} doesn't match existing type.")

        if "recv_nodes" not in convert_dict:
            convert_dict["recv_nodes"] = set()

        convert_dict["recv_nodes"].update(consumer_names)

    def check_nodes_are_not_convert_consumers(self, tensor_name: str, node_names: set[str]):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            return True

        overrides = self.overrides[tensor_name][0]

        if "convert" not in overrides:
            return True

        convert_dict = overrides["convert"]

        if "recv_nodes" not in convert_dict:
            return False

        return not convert_dict["recv_nodes"].intersection(node_names)

    # Required implementations of abstract methods in collections.abc.MutableMapping
    # so that this class can be used like a dict.
    def __setitem__(self, key: str, value: list[dict]):
        self.overrides[key] = value

    def __getitem__(self, key: str) -> list[dict]:
        return self.overrides[key]

    def __delitem__(self, key: str):
        del self.overrides[key]

    def __iter__(self):
        return iter(self.overrides)

    def __len__(self):
        return len(self.overrides)

    def __str__(self) -> str:
        return str(self.overrides)

    def __repr__(self) -> str:
        return f"{super().__repr__()}, TensorQuantOverridesHelper({self.overrides})"


@dataclass
class TensorTypeRequest:
    producer_type: QuantType | None
    consumers: tuple[QuantType, set[str]] | None


class MixedPrecisionTensorQuantOverridesFixer:
    def __init__(
        self,
        overrides: TensorQuantOverridesHelper,
        producers: dict[str, onnx.NodeProto],
        consumers: dict[str, onnx.NodeProto],
        value_infos: dict[str, onnx.ValueInfoProto],
        initializers: dict[str, onnx.TensorProto],
    ):
        self.overrides = overrides
        self.consumers = consumers
        self.producers = producers
        self.value_infos = value_infos
        self.initializers = initializers

    @staticmethod
    def create_from_model(overrides: TensorQuantOverridesHelper, model: onnx.ModelProto):
        consumers = {}
        producers = {}

        # Build dictionaries that map a tensor name to the consumer or producer nodes.
        for node in model.graph.node:
            for input_name in node.input:
                if input_name:
                    if input_name not in consumers:
                        consumers[input_name] = []

                    consumers[input_name].append(node)

            for output_name in node.output:
                producers[output_name] = node

        # Build dictionaries that enable convenient lookups of initializers and value_infos by name.
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})

        return MixedPrecisionTensorQuantOverridesFixer(overrides, producers, consumers, value_infos, initializers)

    def apply(self):
        type_requests = self.get_desired_tensor_types()
        default_activation_qtype = self.overrides.default_activation_qtype

        # Use type requests to "fix" tensor quantization overrides by adding
        # quantization type conversions where necessary.
        for tensor_name, type_req in type_requests.items():
            has_producer_req = type_req.producer_type is not None
            has_consumer_req = bool(type_req.consumers)

            # Only producer type: Add conversion back to default activation type
            if has_producer_req and not has_consumer_req:
                if tensor_name not in self.overrides:
                    self.overrides[tensor_name] = [{"quant_type": type_req.producer_type}]

                self.overrides[tensor_name][0]["convert"] = {"quant_type": default_activation_qtype}
            # Only consumers
            elif not has_producer_req and has_consumer_req:
                prod_type = self.overrides.get_node_output_qtype(tensor_name)
                consumer_type = type_req.consumers[0]

                if prod_type != consumer_type:
                    self.overrides.update_converted_tensor(tensor_name, prod_type, consumer_type, type_req.consumers[1])
                else:
                    if not self.overrides.check_nodes_are_not_convert_consumers(tensor_name, type_req.consumers[1]):
                        raise ValueError(
                            f"Tensor override for '{tensor_name}' converts the type for consumers that need the original type."
                        )
            # Both producer and consumers
            elif has_producer_req and has_consumer_req:
                prod_type = type_req.producer_type
                consumer_type = type_req.consumers[0]

                if prod_type != consumer_type:
                    self.overrides.update_converted_tensor(tensor_name, prod_type, consumer_type, type_req.consumers[1])
                else:
                    all_consumers = set([node.name for node in self.consumers[tensor_name]])
                    consumers_for_original_type = all_consumers.difference(type_req.consumers[1])

                    if len(consumers_for_original_type) == 0:
                        # All consumers want the overridden type, so no need for convert nodes!
                        # Just add the override to the new new if not already present.
                        if tensor_name not in self.overrides:
                            self.overrides[tensor_name] = [{"quant_type": prod_type}]

                        assert "convert" not in self.overrides[tensor_name][0]
                    else:
                        # Some consumers don't want the overridden type.
                        self.overrides.update_converted_tensor(
                            tensor_name, prod_type, default_activation_qtype, consumers_for_original_type
                        )
            else:
                raise ValueError(f"TypeRequest for tensor {tensor_name} has no producer or consumers.")

    def get_desired_tensor_types(self):
        type_requests = {}
        default_activation_qtype = self.overrides.default_activation_qtype

        # Scan tensor overrides for type conversion requests.
        for tensor_name, override_list in self.overrides.items():
            if not self.__is_tensor_quantizable(tensor_name):
                continue  # Skip non-quantizable tensors (e.g., not a float)

            if tensor_name in self.initializers:
                continue  # Skip initializers

            if not override_list or len(override_list) > 1:
                continue  # Skip per-channel stuff

            override = override_list[0]
            quant_type = override.get("quant_type", default_activation_qtype)
            producer_node = self.producers.get(tensor_name)  # None if this is a model input

            if quant_type != default_activation_qtype and "convert" not in override:
                if producer_node is not None:
                    self._build_desired_types_for_node(type_requests, quant_type, producer_node)

                # Find all consumer nodes of `tensor_name` and update their inputs/outputs to the new type.
                for consumer_node in self.consumers[tensor_name]:
                    self._build_desired_types_for_node(type_requests, quant_type, consumer_node)

        return type_requests

    def _build_desired_types_for_node(
        self,
        type_requests: dict[str, TensorTypeRequest],
        quant_type: QuantType,
        node: onnx.NodeProto,
    ):
        # Add output side
        for output_name in node.output:
            if not self.__is_tensor_quantizable(output_name):
                continue

            if output_name not in type_requests:
                type_requests[output_name] = TensorTypeRequest(quant_type, None)
            else:
                if (
                    type_requests[output_name].producer_type is not None
                    and type_requests[output_name].producer_type != quant_type
                ):
                    raise ValueError(f"Tensor {output_name} has multiple types.")

                type_requests[output_name].producer_type = quant_type

        # Add the consumer side
        for input_name in node.input:
            if input_name and input_name not in self.initializers and self.__is_tensor_quantizable(input_name):
                if input_name not in type_requests:
                    type_requests[input_name] = TensorTypeRequest(None, None)

                if type_requests[input_name].consumers is None:
                    type_requests[input_name].consumers = (quant_type, set())

                if type_requests[input_name].consumers[0] != quant_type:
                    raise ValueError(f"Tensor {input_name} has consumers requesting different types.")

                if not node.name:
                    raise ValueError(
                        f"Node of type {node.op_type} with output 0 {node.output[0]} does not have a name!"
                    )

                type_requests[input_name].consumers[1].add(node.name)

    # TODO: This should either be a shared util or should be a closure that is passed in
    # to the constructor.
    def __is_tensor_quantizable(self, tensor_name):
        weight = self.initializers.get(tensor_name)
        if weight is not None:
            if weight.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                return True
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
            ):
                return True

        return False
