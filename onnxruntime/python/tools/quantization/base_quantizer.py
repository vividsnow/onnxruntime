# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import onnx_pb as onnx_proto

try:
    from onnx.reference.op_run import to_array_extended
except ImportError:
    # old version of onnx.
    to_array_extended = None

from .calibrate import TensorData
from .onnx_model import ONNXModel
from .quant_utils import (
    ONNX_TYPE_TO_NP_TYPE,
    TENSOR_NAME_QUANT_SUFFIX,
    QuantType,
    compute_scale_zp,
    compute_scale_zp_float8,
    find_by_name,
    model_has_infer_metadata,
    quantize_data,
    quantize_nparray,
    save_and_reload_model_with_shape_infer,
    tensor_proto_to_array,
)


class BaseQuantizer:
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        if not model_has_infer_metadata(model):
            model = save_and_reload_model_with_shape_infer(model)
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})

        self.model = ONNXModel(model)
        self.per_channel = per_channel  # weight-pack per channel
        self.reduce_range = reduce_range

        self.extra_options = extra_options if extra_options else {}
        self.enable_subgraph_quantization = (
            "EnableSubgraph" in self.extra_options and self.extra_options["EnableSubgraph"]
        )
        self.parent = None
        self.force_quantize_no_input_check = (
            "ForceQuantizeNoInputCheck" in self.extra_options and self.extra_options["ForceQuantizeNoInputCheck"]
        )
        self.is_weight_symmetric = self.extra_options.get(
            "WeightSymmetric", weight_qType in (QuantType.QInt8, QuantType.QInt16, QuantType.QFLOAT8E4M3FN)
        )
        self.is_activation_symmetric = self.extra_options.get("ActivationSymmetric", False)
        self.min_real_range = self.extra_options.get("MinimumRealRange")

        self.activation_qType = getattr(activation_qType, "tensor_type", activation_qType)
        self.weight_qType = getattr(weight_qType, "tensor_type", weight_qType)

        """
            Dictionary specifying the min and max values for tensors. It has following format:
                {
                    "param_name": [min, max]
                }
            example:
                {
                    'Conv_3:0': [np.float32(0), np.float32(0.5)],
                    'Conv_4:0': [np.float32(1), np.float32(3.5)]
                }
        """
        if tensors_range is not None and any(map(lambda t: not isinstance(t, TensorData), tensors_range.values())):
            raise TypeError(
                f"tensors_range contains unexpected types {set(type(v) for v in tensors_range.values())}, not TensorData."
            )
        self.tensors_range = tensors_range
        self.nodes_to_quantize = nodes_to_quantize  # specific nodes to quantize
        self.nodes_to_exclude = nodes_to_exclude  # specific nodes to exclude
        self.op_types_to_quantize = op_types_to_quantize

        self.opset_version = self.check_opset_version()

        # TODO: Remove from on-qdq quantization
        self.tensor_quant_overrides = self._get_and_check_tensor_quant_overrides()

    def quantize_model(self):
        raise NotImplementedError

    def is_input_a_initializer(self, input_name):
        initializer = find_by_name(input_name, self.model.initializer())
        return initializer is not None

    def is_per_channel(self):
        return self.per_channel

    def is_valid_quantize_weight(self, weight_name):
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16)
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            return False
        return self.parent.is_valid_quantize_weight(weight_name)

    def should_quantize_node(self, node):
        if (
            self.nodes_to_quantize is not None
            and len(self.nodes_to_quantize) != 0
            and node.name not in self.nodes_to_quantize
        ):
            return False

        if node.op_type not in self.op_types_to_quantize:
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        return True

    def check_opset_version(self):
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if len(ai_onnx_domain) != 1:
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version == 10:
            logging.warning(
                "The original model opset version is {}, which does not support node fusions. Please update the model to opset >= 11 for better performance.".format(
                    opset_version
                )
            )
            return 10

        if opset_version < 10:
            logging.warning(
                "The original model opset version is {}, which does not support quantization. Please update the model to opset >= 11. Updating the model automatically to opset 11. Please verify the quantized model.".format(
                    opset_version
                )
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        if opset_version < 19 and self.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
            logging.warning(
                "The original model opset version is {}, which does not support quantization to float 8. "
                "Please update the model to opset >= 19. Updating the model automatically to opset 19. "
                "Please verify the quantized model.".format(opset_version)
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 19)])
            self.model.model.ir_version = 9
            opset_version = 19

        return opset_version

    def quantize_bias_static_impl(self, bias_name, input_scale, weight_scale, beta=1.0):
        """
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = tensor_proto_to_array(bias_initializer)
        quantized_bias_name = bias_name + TENSOR_NAME_QUANT_SUFFIX

        # quantize bias
        if self.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
            data = np.asarray(bias_data)
            if data.dtype == np.float16:
                node_qtype = onnx.TensorProto.FLOAT16
            elif data.dtype == np.float32:
                node_qtype = onnx.TensorProto.FLOAT
            else:
                raise TypeError(f"Only float16 or float32 are supported with float 8 but bias dtype is {data.dtype}.")
            quantized_data = data.astype(np.float32)
            bias_scale = np.array([1], dtype=quantized_data.dtype)
            bias_scale_data = bias_scale.reshape(-1)
            packed_bias_initializer = onnx.numpy_helper.from_array(quantized_data, quantized_bias_name)
            self.model.initializer_extend([packed_bias_initializer])
            node_type = "Cast"
        else:
            # calculate scale for bias
            # TODO: This formula should be explained including why the scale is not estimated for the bias as well.
            bias_scale = input_scale * weight_scale * beta

            quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

            # update bias initializer
            bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
            packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
            self.model.initializer_extend([packed_bias_initializer])
            bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
            node_type = "DequantizeLinear"
            node_qtype = self.weight_qType

        # update scale initializer
        quantized_bias_scale_name = quantized_bias_name + "_scale"
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer_extend([packed_bias_scale_initializer])

        # update zero initializer
        if self.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
            tensor_type = self.weight_qType
        else:
            tensor_type = onnx_proto.TensorProto.INT32

        quantized_bias_zp_name = quantized_bias_name + "_zero_point"
        if self.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
            packed_bias_zp_initializer = onnx.helper.make_tensor(quantized_bias_zp_name, self.weight_qType, [1], [0.0])
        elif self.is_per_channel():
            bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
            packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        else:
            packed_bias_zp_initializer = onnx.helper.make_tensor(quantized_bias_zp_name, tensor_type, [], [0])
        self.model.initializer_extend([packed_bias_zp_initializer])

        return (
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            bias_scale_data,
            node_type,
            node_qtype,
        )

    def quantize_initializer_impl(self, weight, qType, reduce_range=False, keep_float_weight=False):
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Quantize weight data. Use quantization overrides if provided by the user.
        weight_data = tensor_proto_to_array(weight)
        quant_overrides = self.get_per_tensor_quant_overrides(weight.name)
        if "quant_type" in quant_overrides:
            qType = quant_overrides["quant_type"].tensor_type  # noqa: N806

        if "scale" in quant_overrides and "zero_point" in quant_overrides:
            zero_point = np.array(quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[qType])
            scale = np.array(quant_overrides["scale"])
            q_weight_data = quantize_nparray(qType, weight_data.flatten(), scale, zero_point)
            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert (
                zero_point.dtype != np.float32 and zero_point.dtype != np.float16
            ), f"Unexpected dtype {zero_point.dtype}"
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        else:
            _, _, zero_point, scale, q_weight_data = quantize_data(
                weight_data.flatten(),
                qType,
                quant_overrides.get("symmetric", self.is_weight_symmetric),
                reduce_range=quant_overrides.get("reduce_range", self.reduce_range and reduce_range),
                min_real_range=self.min_real_range,
                rmin_override=quant_overrides.get("rmin"),
                rmax_override=quant_overrides.get("rmax"),
            )

            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert (
                zero_point.dtype != np.float32 and zero_point.dtype != np.float16
            ), f"Unexpected dtype {zero_point.dtype}"
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        scale_dtype = weight.data_type
        scale_initializer = onnx.helper.make_tensor(scale_name, scale_dtype, [], scale.reshape((-1,)).tolist())
        zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], zero_point.reshape((-1,)).tolist())
        self.model.initializer_extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            if self.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.data_type = self.weight_qType
                q_weight_initializer.dims.extend(weight.dims)
                q_weight_initializer.name = q_weight_name
                # Do not remove .flatten().copy() numpy is not clear about data persistence.
                q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
                if to_array_extended is not None:
                    # This test should not be needed but it helped catch some issues
                    # with data persistence and tobytes.
                    check = to_array_extended(q_weight_initializer)
                    if check.shape != weight_data.shape or check.tobytes() != q_weight_data.tobytes():
                        raise RuntimeError(
                            f"The initializer of shape {weight_data.shape} could not be created, expecting "
                            f"{q_weight_data.tobytes()[:10]}, got {check.tobytes()[:10]} and shape={weight.shape}"
                            f"\nraw={str(q_weight_initializer)[:200]}."
                        )
            else:
                q_weight_data = np.asarray(q_weight_data, dtype=onnx.helper.tensor_dtype_to_np_dtype(qType)).reshape(
                    weight.dims
                )
                q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel_impl(
        self,
        weight_name,
        weight_qType,
        channel_axis,
        reduce_range=True,
        keep_float_weight=False,
    ):
        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        channel_count = weights.shape[channel_axis]
        quant_overrides_for_channels = self.get_per_channel_quant_overrides(weight_name, channel_count)

        # If user provides per-channel quantization overrides, all channels must use the same quantization type.
        # So, just use the first channel's type.
        if "quant_type" in quant_overrides_for_channels[0]:
            weight_qType = quant_overrides_for_channels[0]["quant_type"].tensor_type  # noqa: N806

        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            channel_quant_overrides = quant_overrides_for_channels[i]

            if "scale" in channel_quant_overrides and "zero_point" in channel_quant_overrides:
                zero_point = np.array(channel_quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[weight_qType])
                scale = np.array(channel_quant_overrides["scale"])
                quantized_per_channel_data = quantize_nparray(
                    weight_qType, per_channel_data.flatten(), scale, zero_point
                )
                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert (
                    zero_point.dtype != np.float32 and zero_point.dtype != np.float16
                ), f"Unexpected dtype {zero_point.dtype}"
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(
                    quantized_per_channel_data, np.ndarray
                ), f"Unexpected type {type(quantized_per_channel_data)}"

            else:
                symmetric = channel_quant_overrides.get(
                    "symmetric",
                    (
                        self.is_weight_symmetric
                        or weight_qType in (onnx_proto.TensorProto.INT8, onnx_proto.TensorProto.FLOAT8E4M3FN)
                    ),
                )
                _, _, zero_point, scale, quantized_per_channel_data = quantize_data(
                    per_channel_data.flatten(),
                    weight_qType,
                    symmetric,
                    reduce_range=channel_quant_overrides.get("reduce_range", self.reduce_range and reduce_range),
                    min_real_range=self.min_real_range,
                    rmin_override=channel_quant_overrides.get("rmin"),
                    rmax_override=channel_quant_overrides.get("rmax"),
                )

                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert (
                    zero_point.dtype != np.float32 and zero_point.dtype != np.float16
                ), f"Unexpected dtype {zero_point.dtype}"
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(
                    quantized_per_channel_data, np.ndarray
                ), f"Unexpected type {type(quantized_per_channel_data)}"

            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one
        reshape_dims = list(weights.shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(
            scale_name, initializer.data_type, zero_scale_shape, np.hstack(scale_list).tolist()
        )
        zero_initializer = onnx.helper.make_tensor(
            zp_name, weight_qType, zero_scale_shape, np.hstack(zero_point_list).tolist()
        )

        self.model.initializer_extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights,
                dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight_qType],
            ).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    # TODO: Remove
    def _get_and_check_tensor_quant_overrides(self):
        """
        Get tensor quantization overrides and check correctness.
        """
        tensor_quant_overrides = self.extra_options.get("TensorQuantOverrides", {})

        # Validate that compatible/valid overrides are provided.
        if tensor_quant_overrides:
            initializer_names = self.model.get_initializer_name_set()
            value_info_names = set(self.value_infos.keys())
            keys_unsupported_with_scale_zp = {"symmetric", "reduce_range", "rmax", "rmin"}

            for tensor_name, quant_overrides_list in tensor_quant_overrides.items():
                if tensor_name not in initializer_names and tensor_name not in value_info_names:
                    raise ValueError(f"Tensor '{tensor_name}' in TensorQuantOverrides is not present in the model")

                if not isinstance(quant_overrides_list, list):
                    raise ValueError(f"Tensor quantization overrides for '{tensor_name}' are not in a list")

                is_initializer = tensor_name in initializer_names
                if not is_initializer and len(quant_overrides_list) > 1:
                    raise ValueError(
                        f"Tensor '{tensor_name}' has a list of per-channel overrides, but is not an initializer"
                    )

                quant_type = None
                for index, quant_overrides in enumerate(quant_overrides_list):
                    if not isinstance(quant_overrides, dict):
                        raise ValueError(
                            f"Tensor quantization overrides at index {index} for '{tensor_name}' are not in a dict"
                        )

                    # For per-channel quantization, all channels must use the same quantization type.
                    # Therefore, if the user tries to override the quant_type for a channel, it must match in all
                    # other channels.
                    if index == 0:
                        quant_type = quant_overrides.get("quant_type")
                    elif quant_type != quant_overrides.get("quant_type"):
                        raise ValueError(
                            "Channel quantization types for tensor '{tensor_name}' do not match at index {index}."
                        )

                    has_scale = "scale" in quant_overrides
                    has_zero_point = "zero_point" in quant_overrides

                    if (has_scale and not has_zero_point) or (has_zero_point and not has_scale):
                        raise ValueError(
                            "Must provide both 'scale' and 'zero_point' if one of the overrides is provided"
                        )

                    if has_scale:
                        for key in keys_unsupported_with_scale_zp:
                            if key in quant_overrides:
                                raise ValueError(
                                    f"Tensor override option '{key}' is invalid with 'scale' and 'zero_point'"
                                )

        return tensor_quant_overrides

    # TODO: Remove
    def get_per_tensor_quant_overrides(self, tensor_name):
        quant_overrides_list = self.tensor_quant_overrides.get(tensor_name, [{}])
        num_overrides = len(quant_overrides_list)
        if num_overrides > 1:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to use per-tensor quantization overrides, "
                f"but found {num_overrides} per-channel overrides."
            )

        return quant_overrides_list[0] if num_overrides > 0 else {}

    # TODO: Remove
    def get_per_channel_quant_overrides(self, tensor_name, num_channels):
        quant_overrides_list = self.tensor_quant_overrides.get(tensor_name, [{} for i in range(num_channels)])

        if len(quant_overrides_list) != num_channels:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to have {num_channels} per-channel quantization overrides, "
                f"but found {len(quant_overrides_list)} instead."
            )

        return quant_overrides_list
