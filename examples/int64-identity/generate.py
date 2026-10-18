"""Generate a tiny ONNX model with a non-float tensor input."""

import sys

import onnx
from onnx import TensorProto, helper


model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "int64-identity",
        [helper.make_tensor_value_info("input", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [2])],
    ),
    producer_name="edge-nix",
    opset_imports=[helper.make_opsetid("", 13)],
)
model.ir_version = 10
onnx.save(model, sys.argv[1])
