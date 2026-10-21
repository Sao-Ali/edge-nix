# edge-nix

`edge-nix` uses Nix to package ONNX models for edge inference. A developer supplies a model and build settings, then gets a transferable archive with the prepared model and a manifest. Nix coordinates existing tools and records the build inputs. `edge-nix` does not transfer files to a device.

The current backend accepts a single-file ONNX model, optionally applies ONNX Runtime's dynamic INT8 quantization, and packages it for ONNX Runtime's CPU provider. A recipe can set a maximum model size. The archive is portable, but it does not include an edge-device runtime or support the VENTUNO Q NPU yet. The build output also has host-only inference and validation commands for development.

## What you give it and what you get

You add `edge-nix` as an input to your own Nix flake and pass a path to a single-file `.onnx` model as `modelFile`. There is no upload service: Nix reads that file during the build. Set `quantization` to `"none"` or `"dynamic-int8"`, and optionally set `maxModelBytes` to reject an oversized prepared model. JSON input and expected-output fixtures are optional for packaging but let you check inference in your pipeline.

`nix build` produces `result/edge-package.tar.gz`, containing exactly `model.onnx` and `manifest.json`. The manifest records model hashes, input and output signatures, build settings, target system, and required runtime versions. The archive is the file to transfer to a target with a compatible Python, NumPy, and ONNX Runtime CPU environment. `result/bin/edge-run` and `result/bin/edge-validate` are host-side development commands and are not inside the archive.

This helps when you need a repeatable model preparation step, a traceable artifact, and a CI check that the packaged model still produces expected results before you transfer it to a device. It does not install dependencies or run your application on the device.

## Use edge-nix in your project

[examples/consumer](examples/consumer) is a separate flake with its own `model.onnx` and validation fixtures. From that directory, build it against this checkout:

```sh
cd examples/consumer
nix build --override-input edgeNix path:../.. --no-write-lock-file
tar -tzf result/edge-package.tar.gz
./result/bin/edge-validate @sample.json @expected.json
```

In your own project, add `edgeNix` as a flake input and call `edgeNix.lib.mkOnnxBundle pkgs` as shown in [the consumer flake](examples/consumer/flake.nix). Replace its `model.onnx`, `sample.json`, and `expected.json` with your files. Run `nix build` to make the archive and `nix flake check` to run validation in your pipeline. You do not need a connected device for either command.

The file to transfer is `result/edge-package.tar.gz`. It contains only regular files named `model.onnx` and `manifest.json`. The `result/bin` commands and the convenience links at the root of `result` refer to the build host's Nix store; do not copy `result` as a device package. The manifest records the target system and required Python, NumPy, and ONNX Runtime versions. Those runtime packages are not inside the archive, so the target must provide compatible dependencies.

## Build the included MNIST example

On Linux or Apple Silicon macOS with flakes enabled:

```sh
nix build .#demo-fp32
./result/bin/edge-run @examples/mnist/sample.json
./result/bin/edge-validate @examples/mnist/sample.json @examples/mnist/expected.json

nix build .#demo-int8
./result/bin/edge-run @examples/mnist/sample.json
./result/bin/edge-validate @examples/mnist/sample.json @examples/mnist/expected.json
```

Both `edge-run` commands return ten scores under `Plus214_Output_0`; the highest score should be at index `7`, the digit in the sample image. `edge-validate` extracts the archive, checks its files and model hash, runs inference, checks the expected output shape and class, and prints a JSON report. Packages are defined for `x86_64-linux`, `aarch64-linux`, and `aarch64-darwin` so the CPU path can also be checked during development on a Mac.

The model is fetched from a pinned upstream revision with a verified hash in [examples/mnist/model.nix](examples/mnist/model.nix). `sample.json` contains the first image from the [MNIST test set](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz), scaled to the `[0, 1]` range required by the model. Its known label is `7`. The sample shows that both variants can classify a real image; it is not a broad accuracy or performance evaluation.

The bundle function accepts these settings in another flake:

```nix
let
  mkOnnxBundle = edgeNix.lib.mkOnnxBundle pkgs;
in
mkOnnxBundle {
  name = "my-edge-model";
  modelFile = ./model.onnx;
  quantization = "dynamic-int8"; # or "none"
  maxModelBytes = 1048576; # optional artifact size limit
}
```

The `edge-run` command is a host-only smoke-test interface: pass a JSON object whose keys match the model's input names, or use `@input.json` to read them from a file. The CPU backend accepts float16/32/64, signed and unsigned 8/16/32/64-bit integer, and boolean tensor inputs. `maxModelBytes` bounds the packaged model file only; it does not predict peak RAM use. Models with ONNX external data files are rejected.

`manifest.json` version 2 records the source and packaged model sizes and SHA-256 hashes, input and output names, types and shapes, quantization setting, runtime requirements, and target system. The build checks that the ONNX graph is valid and loadable. The archive contains no Nix store paths or symlinks.

For another model, provide `edge-validate` with an input JSON object and an expectation JSON object. Each entry under `outputs` can specify `shape`, flattened `argmax`, or `values`. Values match exactly unless `atol` or `rtol` is supplied. See [examples/mnist/expected.json](examples/mnist/expected.json). To compare two archives on the same input, add `--reference-archive path/to/reference.tar.gz --atol 0.25 --rtol 0.01`; the report includes the maximum absolute output difference and fails if the tolerances are exceeded.

## Automated checks

```sh
nix build .#checks.aarch64-darwin.demo-fp32 \
  .#checks.aarch64-darwin.demo-int8 \
  .#checks.aarch64-darwin.demo-int64
# On Linux, use checks.x86_64-linux or checks.aarch64-linux instead.
```

The INT8 check also compares its scores with the FP32 model on the sample image. These fixtures verify one input, not broad model accuracy or memory use.

`demo-int64` is a generated two-element integer identity model. Its check exercises the same bundle and validator with a non-float input and exact output values.

GitHub Actions runs these three checks and the separate consumer flake check on x86_64 Linux, aarch64 Linux, and Apple Silicon macOS for pushes and pull requests. The VM test remains a separate Linux check.

## VM test

```sh
nix build .#checks.x86_64-linux.vm
```

The NixOS VM test boots a Linux guest, runs both example variants, and checks their output. It requires a host or remote builder able to run that VM. A VM verifies loading and inference; actual accelerator use, latency, memory, and power must be tested on the target board.

## Next target

The intended first hardware target is the Arduino VENTUNO Q. Its Qualcomm accelerator needs a different export and runtime path from this CPU package. A future backend must package that target-specific output and pass on-board tests before `edge-nix` can claim VENTUNO Q NPU support.

## License

The edge-nix source code and documentation are available under the [MIT License](LICENSE). You remain responsible for the license of any model you package.
