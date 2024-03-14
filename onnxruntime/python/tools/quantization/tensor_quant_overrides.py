# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import onnx

from .quant_utils import QuantType


@dataclass
class QuantTypeInfo:
    quant_type: QuantType
    symmetric: bool | None = None
    reduce_range: bool | None = None

    def __eq__(self, other: object):
        if isinstance(other, QuantTypeInfo):
            return (
                self.quant_type == other.quant_type
                and (self.symmetric is None or other.symmetric is None or self.symmetric == other.symmetric)
                and (self.reduce_range is None or other.reduce_range is None or self.reduce_range == other.reduce_range)
            )
        return NotImplemented

    @staticmethod
    def load_from_dict(
        raw_dict: dict[str, Any],
        default_activation_qtype: QuantType | None = None,
        default_activation_symmetric: bool | None = None,
        default_activation_reduce_range: bool | None = None,
    ) -> QuantTypeInfo:
        return QuantTypeInfo(
            raw_dict.get("quant_type", default_activation_qtype),
            raw_dict.get("symmetric", default_activation_symmetric),
            raw_dict.get("reduce_range", default_activation_reduce_range),
        )

    def save_to_dict(self, raw_dict: dict[str, Any]):
        raw_dict["quant_type"] = self.quant_type
        if self.symmetric is not None:
            raw_dict["symmetric"] = self.symmetric
        if self.reduce_range is not None:
            raw_dict["reduce_range"] = self.reduce_range


class TensorQuantOverridesHelper(MutableMapping):
    def __init__(self, raw_overrides: dict[str, list[dict[str, Any]]]):
        self.overrides = raw_overrides
        self.quant_types = None

    def get_per_tensor_overrides(self, tensor_name: str) -> dict[str, Any]:
        overrides_list = self.overrides.get(tensor_name, [{}])
        num_overrides = len(overrides_list)
        if num_overrides > 1:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to use per-tensor quantization overrides, "
                f"but found {num_overrides} per-channel overrides."
            )

        return overrides_list[0] if num_overrides > 0 else {}

    def get_per_channel_overrides(
        self,
        tensor_name: str,
        num_channels: int,
    ) -> list[dict[str, Any]]:
        overrides_list = self.overrides.get(tensor_name, [{} for i in range(num_channels)])

        if len(overrides_list) != num_channels:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to have {num_channels} per-channel quantization overrides, "
                f"but found {len(overrides_list)} instead."
            )

        return overrides_list

    def get_node_output_qtype_info(
        self,
        output_name: str,
        default_qtype: QuantType | None,
        default_symmetric: bool | None = None,
    ) -> QuantTypeInfo:
        if output_name not in self.overrides:
            return QuantTypeInfo(default_qtype, default_symmetric)

        # Get the first overrides dict in the list. This works for both per-tensor and per-channel
        # quantization because all channels must use the same quant type.
        tensor_overrides = self.overrides[output_name][0]

        return QuantTypeInfo(
            tensor_overrides.get("quant_type", default_qtype),
            tensor_overrides.get("symmetric", default_symmetric),
        )

    def get_node_input_qtype_info(
        self,
        input_name: str,
        node_name: str,
        default_qtype: QuantType | None,
        default_symmetric: bool | None = None,
        default_reduce_range: bool | None = None,
    ) -> QuantTypeInfo:
        if input_name not in self.overrides or not self.overrides[input_name]:
            return QuantTypeInfo(default_qtype, default_symmetric, default_reduce_range)

        # Get the first overrides dict in the list. This works for both per-tensor and per-channel
        # quantization because all channels must use the same quant type.
        tensor_overrides = self.overrides[input_name][0]
        producer_type = tensor_overrides.get("quant_type", default_qtype)

        if "convert" not in tensor_overrides:
            return QuantTypeInfo(producer_type, default_symmetric, default_reduce_range)

        # This tensor is converted. Check if the node gets the original qtype or the converted qtype.
        convert_dict = tensor_overrides["convert"]
        qtype_info = QuantTypeInfo(
            producer_type,
            convert_dict.get("symmetric", default_symmetric),
            convert_dict.get("reduce_range", default_reduce_range),
        )

        # Check if all nodes receive the coverted type (i.e., recv_nodes is None) or this node
        # is in the list of consumers (recv_nodes).
        if ("recv_nodes" not in convert_dict) or (node_name in convert_dict["recv_nodes"]):
            qtype_info.quant_type = convert_dict["quant_type"]

        return qtype_info

    def update_tensor_overrides(
        self,
        tensor_name: str,
        new_vals: dict[str, Any],
        channels: list[int] | None = None,
        overwrite: bool = True,
    ) -> bool:
        if not new_vals:
            return False

        channels = set(channels) if channels is not None else None
        have_overrides = self.overrides.get(tensor_name)

        # If `overwrite` is False, check if we would overwrite anything.
        do_update = True
        if not overwrite and have_overrides:
            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                if set(new_vals).intersection(set(overrides)):
                    do_update = False
                    break

        # Do the update if `overwrite` is True or if nothing is overwritten (do not want partial overwrites).
        if do_update:
            if not have_overrides:
                self.overrides[tensor_name] = [{}]

            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                overrides.update(new_vals)

        return do_update

    def get_quant_types(self) -> set[QuantType]:
        if self.quant_types is not None:
            return self.quant_types

        self.quant_types = set()

        if self.overrides:
            for quant_overrides_list in self.overrides.values():
                for quant_overrides in quant_overrides_list:
                    if "quant_type" in quant_overrides:
                        self.quant_types.add(quant_overrides["quant_type"])

                    if "convert" in quant_overrides and "quant_type" in quant_overrides["convert"]:
                        self.quant_types.add(quant_overrides["convert"]["quant_type"])

        return self.quant_types

    def is_valid(
        self,
        initializer_names: set[str],
        activation_names: set[str],
        default_activation_qtype,
    ) -> tuple[bool, str | None]:
        self.quant_types = set()

        # Validate that compatible/valid overrides are provided.
        if self.overrides:
            keys_unsupported_with_scale_zp = {"symmetric", "reduce_range", "rmax", "rmin"}

            for tensor_name, quant_overrides_list in self.overrides.items():
                if tensor_name not in initializer_names and tensor_name not in activation_names:
                    return False, f"Tensor '{tensor_name}' in TensorQuantOverrides is not present in the model"

                if not isinstance(quant_overrides_list, list):
                    return False, f"Tensor quantization overrides for '{tensor_name}' are not in a list"

                is_initializer = tensor_name in initializer_names
                if not is_initializer and len(quant_overrides_list) > 1:
                    return (
                        False,
                        f"Tensor '{tensor_name}' has a list of per-channel overrides, but is not an initializer",
                    )

                quant_type = None
                for index, quant_overrides in enumerate(quant_overrides_list):
                    if not isinstance(quant_overrides, dict):
                        return (
                            False,
                            f"Tensor quantization overrides at index {index} for '{tensor_name}' are not in a dict",
                        )

                    # For per-channel quantization, all channels must use the same quantization type.
                    # Therefore, if the user tries to override the quant_type for a channel, it must match in all
                    # other channels.
                    if index == 0:
                        quant_type = quant_overrides.get("quant_type")
                        if quant_type:
                            self.quant_types.add(quant_type)
                    elif quant_type != quant_overrides.get("quant_type"):
                        return (
                            False,
                            "Channel quantization types for tensor '{tensor_name}' do not match at index {index}.",
                        )

                    has_scale = "scale" in quant_overrides
                    has_zero_point = "zero_point" in quant_overrides

                    if (has_scale and not has_zero_point) or (has_zero_point and not has_scale):
                        return (
                            False,
                            "Must provide both 'scale' and 'zero_point' if one of the overrides is provided",
                        )

                    if has_scale:
                        for key in keys_unsupported_with_scale_zp:
                            if key in quant_overrides:
                                return (
                                    False,
                                    f"Tensor override option '{key}' is invalid with 'scale' and 'zero_point'",
                                )

                    if "reduce_range" in quant_overrides and not is_initializer:
                        return (
                            False,
                            f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
                        )

                    if "convert" in quant_overrides:
                        if index > 0:
                            return (
                                False,
                                f"Per-channel overrides (tensor '{tensor_name}') do not support 'convert'.",
                            )

                        if is_initializer:
                            return False, "Cannot use 'convert' override for initializers"

                        if "quant_type" not in quant_overrides["convert"]:
                            return False, f"'convert' options (tensor '{tensor_name}') must specify a 'quant_type'"

                        if "reduce_range" in quant_overrides["convert"]:
                            return (
                                False,
                                f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
                            )

                        convert_quant_type = quant_overrides["convert"]["quant_type"]
                        original_quant_type = quant_type if quant_type is not None else default_activation_qtype
                        if convert_quant_type == original_quant_type:
                            return (
                                False,
                                f"'convert' quant_type must differ from original quant_type (tensor '{tensor_name}')",
                            )

                        convert_has_scale = "scale" in quant_overrides["convert"]
                        convert_has_zero_point = "zero_point" in quant_overrides["convert"]

                        if (convert_has_scale and not convert_has_zero_point) or (
                            convert_has_zero_point and not convert_has_scale
                        ):
                            return (
                                False,
                                f"Must provide both 'scale' and 'zero_point' if one of the overrides is provided (tensor '{tensor_name}')",
                            )

                        if convert_has_scale:
                            for key in keys_unsupported_with_scale_zp:
                                if key in quant_overrides["convert"]:
                                    return (
                                        False,
                                        f"Tensor override option '{key}' is invalid with 'scale' and 'zero_point' (tensor '{tensor_name}')",
                                    )

                        self.quant_types.add(convert_quant_type)

        return True, None

    def pprint_str(self, indent=None) -> str:
        return json.dumps(self.overrides, default=str, indent=indent)

    def empty(self) -> bool:
        return len(self.overrides) > 0

    def get_dict(self) -> dict[str, list[dict[str, Any]]]:
        return self.overrides

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
    producer: QuantTypeInfo | None
    consumers: tuple[QuantTypeInfo, set[str]] | None


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

    def apply(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ):
        type_requests = self.get_desired_tensor_types(default_activation_qtype, default_activation_symmetric)

        # Use type requests to "fix" tensor quantization overrides by adding
        # quantization type conversions where necessary.
        for tensor_name, type_req in type_requests.items():
            all_consumers = set([node.name for node in self.consumers.get(tensor_name, [])])
            has_producer_req = type_req.producer is not None
            has_consumer_req = bool(type_req.consumers)

            # Only producer type: Add conversion back to default activation type
            if has_producer_req and not has_consumer_req:
                self._update_converted_tensor(
                    tensor_name, type_req.producer, QuantTypeInfo(default_activation_qtype), all_consumers
                )
            # Only consumers
            elif not has_producer_req and has_consumer_req:
                prod_type_info = self.overrides.get_node_output_qtype_info(tensor_name, default_activation_qtype)
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    if not self._check_nodes_are_not_convert_consumers(tensor_name, type_req.consumers[1]):
                        raise ValueError(
                            f"Tensor override for '{tensor_name}' converts the type for consumers that need the original type."
                        )
            # Both producer and consumers
            elif has_producer_req and has_consumer_req:
                prod_type_info = type_req.producer
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    consumers_for_original_type = all_consumers.difference(type_req.consumers[1])

                    if len(consumers_for_original_type) == 0:
                        # All consumers want the overridden type, so no need for convert nodes!
                        # Just add the override to the new new if not already present.
                        if tensor_name not in self.overrides:
                            self.overrides[tensor_name] = [{}]
                            prod_type_info.save_to_dict(self.overrides[tensor_name][0])

                        assert "convert" not in self.overrides[tensor_name][0]
                    else:
                        # Some consumers don't want the overridden type.
                        self._update_converted_tensor(
                            tensor_name,
                            prod_type_info,
                            QuantTypeInfo(default_activation_qtype),
                            consumers_for_original_type,
                        )
            else:
                raise ValueError(f"TypeRequest for tensor {tensor_name} has no producer or consumers.")

    def get_desired_tensor_types(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ) -> dict[str, TensorTypeRequest]:
        type_requests = {}
        default_activation_type_info = QuantTypeInfo(default_activation_qtype, default_activation_symmetric)

        # Scan tensor overrides for type conversion requests.
        for tensor_name, override_list in self.overrides.items():
            if not self.__is_tensor_quantizable(tensor_name):
                continue  # Skip non-quantizable tensors (e.g., not a float)

            if tensor_name in self.initializers:
                continue  # Skip initializers

            if not override_list or len(override_list) > 1:
                continue  # Skip per-channel stuff

            override_dict = override_list[0]
            quant_type_info = QuantTypeInfo.load_from_dict(override_dict, default_activation_type_info.quant_type)
            producer_node = self.producers.get(tensor_name)  # None if this is a model input

            if quant_type_info != default_activation_type_info and "convert" not in override_dict:
                if producer_node is not None:
                    self._add_type_requests_for_node(type_requests, quant_type_info, producer_node)

                # Find all consumer nodes of `tensor_name` and update their inputs/outputs to the new type.
                for consumer_node in self.consumers.get(tensor_name, []):
                    self._add_type_requests_for_node(type_requests, quant_type_info, consumer_node)

        return type_requests

    def _add_type_requests_for_node(
        self,
        type_requests: dict[str, TensorTypeRequest],
        quant_type_info: QuantTypeInfo,
        node: onnx.NodeProto,
    ):
        # Add output side
        for output_name in node.output:
            if not self.__is_tensor_quantizable(output_name):
                continue

            if output_name not in type_requests:
                type_requests[output_name] = TensorTypeRequest(quant_type_info, None)
            else:
                if (
                    type_requests[output_name].producer is not None
                    and type_requests[output_name].producer != quant_type_info
                ):
                    raise ValueError(f"Tensor {output_name} has multiple types.")

                type_requests[output_name].producer = quant_type_info

        # Add the consumer side
        for input_name in node.input:
            if input_name and input_name not in self.initializers and self.__is_tensor_quantizable(input_name):
                if input_name not in type_requests:
                    type_requests[input_name] = TensorTypeRequest(None, None)

                if type_requests[input_name].consumers is None:
                    type_requests[input_name].consumers = (quant_type_info, set())

                if type_requests[input_name].consumers[0] != quant_type_info:
                    raise ValueError(f"Tensor {input_name} has consumers requesting different types.")

                if not node.name:
                    raise ValueError(
                        f"Node of type {node.op_type} with output 0 {node.output[0]} does not have a name!"
                    )

                type_requests[input_name].consumers[1].add(node.name)

    def _update_converted_tensor(
        self,
        tensor_name: str,
        producer_type_info: QuantTypeInfo,
        consumer_type_info: QuantTypeInfo,
        consumer_names: set[str],
    ):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            self.overrides[tensor_name] = [{}]
            producer_type_info.save_to_dict(self.overrides[tensor_name][0])

        overrides = self.overrides[tensor_name][0]
        if producer_type_info != QuantTypeInfo.load_from_dict(overrides):
            raise ValueError(f"Desired producer quant_type for {tensor_name} doesn't match existing type.")

        if consumer_names:
            if "convert" not in overrides:
                overrides["convert"] = {}
                consumer_type_info.save_to_dict(overrides["convert"])

            convert_dict = overrides["convert"]
            if consumer_type_info != QuantTypeInfo.load_from_dict(convert_dict):
                raise ValueError(f"Desired consumer quant_type for {tensor_name} doesn't match existing type.")

            if "recv_nodes" not in convert_dict:
                convert_dict["recv_nodes"] = set()

            convert_dict["recv_nodes"].update(consumer_names)

    def _check_nodes_are_not_convert_consumers(self, tensor_name: str, node_names: set[str]):
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            return True

        overrides = self.overrides[tensor_name][0]

        if "convert" not in overrides:
            return True

        convert_dict = overrides["convert"]

        if "recv_nodes" not in convert_dict:
            return False

        return not convert_dict["recv_nodes"].intersection(node_names)

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
