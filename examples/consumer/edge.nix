{
  name = "consumer-uno-q";
  model = ./model.onnx;

  target = {
    device = "arduino-uno-q";
    accelerator = "cpu";
  };

  runtime.backend = "onnxruntime";

  quantization.enable = false;

  validation = {
    sampleInput = ./sample.json;
    expectedOutput = ./expected.json;
  };
}
