#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import onnx

from onnxruntime.quantization import QuantType
from onnxruntime.quantization.tensor_quant_overrides import (
    MixedPrecisionTensorQuantOverridesFixer,
    TensorQuantOverridesHelper,
)


class TestMixedPrecisionQuantOverridesFixer(unittest.TestCase):
    def build_test_model_1(self, shape):
        input_0 = onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, shape)
        input_1 = onnx.helper.make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, shape)
        output_0 = onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, shape)
        output_1 = onnx.helper.make_tensor_value_info("output_1", onnx.TensorProto.FLOAT, shape)
        output_2 = onnx.helper.make_tensor_value_info("output_2", onnx.TensorProto.FLOAT, shape)

        op1_node = onnx.helper.make_node("Sigmoid", ["input_0"], ["op1_out"], name="op1")
        op2_node = onnx.helper.make_node("Cos", ["input_1"], ["op2_out"], name="op2")
        op3_node = onnx.helper.make_node("Sin", ["op1_out"], ["op3_out"], name="op3")
        op4_node = onnx.helper.make_node("Tanh", ["op2_out"], ["op4_out"], name="op4")
        op5_node = onnx.helper.make_node("Mul", ["op3_out", "op4_out"], ["op5_out"], name="op5")
        op6_node = onnx.helper.make_node("Relu", ["op5_out"], ["output_0"], name="op6")
        op7_node = onnx.helper.make_node("Cos", ["op2_out"], ["output_1"], name="op7")
        op8_node = onnx.helper.make_node("Sigmoid", ["op2_out"], ["output_2"], name="op8")

        graph = onnx.helper.make_graph(
            [
                op1_node,
                op2_node,
                op3_node,
                op4_node,
                op5_node,
                op6_node,
                op7_node,
                op8_node,
            ],
            "mixed_prec_test",
            [input_0, input_1],
            [output_0, output_1, output_2],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        onnx.checker.check_model(model, True)  # TODO: Remove
        return onnx.shape_inference.infer_shapes(model)

    def test_fixer_1(self):
        shape = (1, 2, 3)
        model = self.build_test_model_1(shape)
        onnx.save_model(model, "model.onnx")

        default_act_qtype = QuantType.QUInt8
        default_wgt_qtype = QuantType.QUInt8
        raw_overrides = {"op4_out": [{"quant_type": QuantType.QUInt16}]}
        overrides = TensorQuantOverridesHelper(raw_overrides, default_act_qtype, default_wgt_qtype)
        fixer = MixedPrecisionTensorQuantOverridesFixer.create_from_model(overrides, model)
        fixer.apply()

        expected = {
            "op4_out": [{"quant_type": QuantType.QUInt16}],
            "op2_out": [
                {"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op4"}}}
            ],
            "op5_out": [
                {"quant_type": QuantType.QUInt16, "convert": {"quant_type": QuantType.QUInt8, "recv_nodes": {"op6"}}}
            ],
            "op3_out": [
                {"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"op5"}}}
            ],
        }
        self.assertEqual(overrides.get_dict(), expected)
