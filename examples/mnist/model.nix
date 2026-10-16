{ pkgs }:
pkgs.fetchurl {
  url = "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5/mnist-12.onnx";
  hash = "sha256-XGiGkPi6z2Z9TCB0r1rQZGyjKNerA+zPlEplsyAXG90=";
}
