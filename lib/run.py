"""Run an ONNX model with JSON inputs; intended as a simple smoke-test interface."""

import json
import sys

import numpy as np
import onnxruntime as ort


DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: edge-run '<JSON inputs>' or edge-run @input.json")

    session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
    argument = sys.argv[2]
    if argument.startswith("@"):
        with open(argument[1:], encoding="utf-8") as handle:
            supplied = json.load(handle)
    else:
        supplied = json.loads(argument)
    if not isinstance(supplied, dict):
        raise SystemExit("inputs must be a JSON object keyed by model input name")

    inputs = {}
    for item in session.get_inputs():
        if item.type not in DTYPES:
            raise SystemExit(f"unsupported input type: {item.type}")
        if item.name not in supplied:
            raise SystemExit(f"missing input: {item.name}")
        inputs[item.name] = np.asarray(supplied[item.name], dtype=DTYPES[item.type])

    values = session.run(None, inputs)
    print(json.dumps({item.name: value.tolist() for item, value in zip(session.get_outputs(), values)}))


if __name__ == "__main__":
    main()
