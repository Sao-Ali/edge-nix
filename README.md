# edge-nix

`edge-nix` is an early experiment in using Nix to prepare models for edge inference. Nix coordinates existing tools and records the inputs to a build. `edge-nix` does not implement a model converter or quantizer itself.

The first backend accepts an ONNX model, optionally applies ONNX Runtime's dynamic INT8 quantization, and produces a model, a manifest, and a CPU inference command. A recipe can set a maximum artifact size, an early edge constraint. The example uses the pretrained [ONNX Model Zoo MNIST classifier](https://huggingface.co/onnxmodelzoo/mnist-12), which recognizes handwritten digits. This is a starting point for device-specific backends, not a VENTUNO Q accelerator build.

## Build the example

On Linux or Apple Silicon macOS with flakes enabled:

```sh
nix build .#demo-fp32
./result/bin/edge-run @examples/mnist/sample.json
./result/bin/edge-validate @examples/mnist/sample.json @examples/mnist/expected.json

nix build .#demo-int8
./result/bin/edge-run @examples/mnist/sample.json
./result/bin/edge-validate @examples/mnist/sample.json @examples/mnist/expected.json
```

Both `edge-run` commands return ten scores under `Plus214_Output_0`; the highest score should be at index `7`, the digit in the sample image. `edge-validate` checks the built model's size and SHA-256 against its manifest, runs inference, checks the expected output shape and class, and prints a JSON report. The output directory also contains `model.onnx` and `manifest.json`. Packages are defined for `x86_64-linux`, `aarch64-linux`, and `aarch64-darwin` so the CPU path can also be checked during development on a Mac.

The model is fetched from a pinned upstream revision with a verified hash in [examples/mnist/model.nix](examples/mnist/model.nix). `sample.json` contains the first image from the [MNIST test set](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz), scaled to the `[0, 1]` range required by the model. Its known label is `7`. The sample shows that both variants can classify a real image; it is not a broad accuracy or performance evaluation.

Use an existing ONNX file in another flake:

```nix
let
  mkOnnxBundle = edge-nix.lib.mkOnnxBundle pkgs;
in
mkOnnxBundle {
  name = "my-edge-model";
  modelFile = ./model.onnx;
  quantization = "dynamic-int8"; # or "none"
  maxModelBytes = 1048576; # optional artifact size limit
}
```

The `edge-run` command is a smoke-test interface: pass a JSON object whose keys match the model's input names, or use `@input.json` to read them from a file. The first backend accepts float16/32/64, signed and unsigned 8/16/32/64-bit integer, and boolean tensor inputs, and uses ONNX Runtime's CPU execution provider. `maxModelBytes` bounds the model file only; it does not predict peak RAM use.

`manifest.json` records the built model's byte size, SHA-256, input and output names, types and shapes, source model path, quantization setting, runtime version, and target system. The build checks that the ONNX graph is valid and loadable. The manifest describes this CPU bundle; it does not claim that another execution provider can run the model.

For another model, provide `edge-validate` with an input JSON object and an expectation JSON object. Each entry under `outputs` can specify `shape`, flattened `argmax`, or `values`. Values match exactly unless `atol` or `rtol` is supplied. See [examples/mnist/expected.json](examples/mnist/expected.json). To compare a bundle against another ONNX model on the same input, add `--reference-model path/to/reference.onnx --atol 0.25 --rtol 0.01`; the report includes the maximum absolute output difference and fails if the tolerances are exceeded.

## Automated checks

```sh
nix build .#checks.aarch64-darwin.demo-fp32 \
  .#checks.aarch64-darwin.demo-int8 \
  .#checks.aarch64-darwin.demo-int64
# On Linux, use checks.x86_64-linux or checks.aarch64-linux instead.
```

The INT8 check also compares its scores with the FP32 model on the sample image. These fixtures verify one input, not broad model accuracy or memory use.

`demo-int64` is a generated two-element integer identity model. Its check exercises the same bundle and validator with a non-float input and exact output values.

GitHub Actions runs these three checks on x86_64 Linux, aarch64 Linux, and Apple Silicon macOS for pushes and pull requests. The VM test remains a separate Linux check.

## VM test

```sh
nix build .#checks.x86_64-linux.vm
```

The NixOS VM test boots a Linux guest, runs both example variants, and checks their output. It requires a host or remote builder able to run that VM. A VM verifies loading and inference; actual accelerator use, latency, memory, and power must be tested on the target board.

## Next target

The intended first hardware target is the Arduino VENTUNO Q. Its Qualcomm accelerator uses a different export and runtime path from this CPU example. Once the board is available, a backend can wrap that upstream path and add hardware tests while retaining the same idea of a declarative model build.
