"""Apply ONNX Runtime's existing dynamic INT8 quantizer."""

import sys

from onnxruntime.quantization import QuantType, quantize_dynamic


if __name__ == "__main__":
    quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QInt8)
