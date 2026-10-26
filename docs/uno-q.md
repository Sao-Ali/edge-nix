# Arduino UNO Q inference bundle candidate

This path runs ONNX Runtime's CPU provider on the QRB2210 Debian Linux MPU. It does not use the Qualcomm GPU, DSP, NPU, QNN, or the STM32 MCU. The archive includes a native AArch64 runner, ONNX Runtime, its ELF dependency closure, a bundled glibc loader, model, manifest, and validation fixture. Nix, Python, and ONNX Runtime installation are not required on the board.

**Status:** clean ARM64 Debian container and one physical UNO Q MNIST smoke test passed. Resource measurements and broader model coverage remain pending. See [the environment record](uno-q-environment.md).

## Build and transfer

Use a native `aarch64-linux` Nix builder. An x86_64 Linux or macOS build of this output is rejected; using a configured remote ARM64 builder is possible by building on that builder. From this repository:

```sh
nix build .#uno-q-demo
tar -tzf result/edge-inference.tar.gz | head
sha256sum -c result/edge-inference.tar.gz.sha256
scp result/edge-inference.tar.gz arduino@192.168.1.247:/tmp/
```

For another model, use `edgeNix.lib.mkInferenceBundle pkgs` on an ARM64 Linux package set:

```nix
edgeNix.lib.mkInferenceBundle pkgs {
  name = "my-uno-q-model";
  modelFile = ./model.onnx;
  target = "arduino-uno-q";
  backend = "onnxruntime-cpu";
  quantization = "none"; # dynamic-int8 also accepted
  sampleInput = ./sample.json;
  expectedOutput = ./expected.json;
}
```

For a separate configuration file, use `edgeNix.lib.mkEdgeSystem pkgs
(import ./edge.nix)` in your consumer flake. See the complete
[consumer configuration](../examples/consumer/edge.nix) and
[flake wiring](../examples/consumer/flake.nix). The configuration requires a
name, model, `target.device = "arduino-uno-q"`, `target.accelerator = "cpu"`,
`runtime.backend = "onnxruntime"`, and validation input and expected-output
paths. `quantization.enable = true` selects dynamic INT8; `type =
"dynamic-int8"` is the only supported type. `maxModelBytes` is optional.
The builder validates the fixture and model during every build. The
configuration file is imported by the consumer flake; `nix build` does not
discover it automatically.

The first runner accepts rectangular numeric JSON tensors with ONNX `tensor(float)` or `tensor(int64)` inputs and outputs. Shapes must match each fixed model dimension; dynamic dimensions are allowed. JSON int64 values must be exact integers within ±(2^53−1). Input keys must match ONNX input names. Expected output JSON supports `shape`, flattened `argmax`, and `values` with optional `atol` and `rtol`, as in [the MNIST fixture](../examples/mnist/expected.json). Models using external ONNX data are rejected during preparation.

## Run on the board

```sh
ssh arduino@192.168.1.247
mkdir -p ~/edge-inference
cd ~/edge-inference
tar -xzf /tmp/edge-inference.tar.gz
cd edge-inference
./bin/edge-run --check
./bin/edge-run --input validation/sample.json
./bin/edge-validate --input validation/sample.json --expected validation/expected.json
```

`--check` verifies every packaged file's SHA-256 digest, AArch64 architecture, Debian 13 userland, Linux kernel 6.1 or newer, and ONNX Runtime model initialization. Validation prints JSON with `passed: true` and output checks, or a JSON error and nonzero exit status. The MNIST fixture is a 28×28 image of digit 7; the expected class is index 7.

The manifest records schema 3, artifact type, model hashes and signatures, target, runtime versions, pinned nixpkgs revision, recipe source hashes, a dependency inventory and per-file hashes. The archive checksum is stored beside the archive to avoid a self-referential hash. The V1 model-only archive and manifest version 2 remain separate.

## Compatibility and troubleshooting

- `Exec format error` or `requires aarch64 Linux`: check `uname -m`; the MPU must report `aarch64`. Do not run the archive on the STM32 MCU.
- `requires Linux kernel 6.1 or newer`: inspect `uname -r`. The tested UNO Q reported `6.16.7-g0dd6551ae96b`.
- `error while loading shared libraries`: inspect `lib/` against `manifest.json`; rerun `--check`. Transfer and extract the whole archive without removing files.
- `bundle hash mismatch`: restore the original archive; a model, runner, library, config or fixture changed after packaging.
- `input rank mismatch`, `input shape mismatch`, or `unsupported input dtype`: supply a JSON tensor matching the input names, dtypes and shapes in `manifest.json`.
- `output ... differs`: check the fixture and its numerical tolerances; `edge-validate` exits nonzero.

The shell launcher uses the packaged loader and library directory. ELF interpreter, `DT_NEEDED`, RPATH and AArch64 class are checked during packaging. Some upstream ELF files retain embedded Nix store strings in diagnostic or provider-path data; their counts are recorded in the manifest. Clean Debian and physical UNO Q execution showed they did not require a Nix store for this CPU example. Arbitrary models may exercise other ONNX Runtime code paths, so each bundle runs a model smoke test during build and should be tested on the board.

For build-side integration, run `nix build .#checks.aarch64-linux.uno-q-demo`; on an ARM64 Linux host with Docker, run `tests/clean-debian.sh result/edge-inference.tar.gz`. The latter uses a network-disabled Debian 13 container and checks inference, corruption, missing library, bad input shape, and wrong expected output. The container check is not a physical board check.

## Measurements

For the current MNIST candidate, the 2026-10-22 ARM64 Docker build produced a 14,257,017-byte compressed archive, a roughly 44 MiB extracted directory, and a 26,143-byte ONNX model. Its archive SHA-256 is `4ff12a1c924964f387ab5fe93219a488076ab870dda7ee434816c94dc2a0ff54`. These are build and container measurements; `result/size.txt` and `result/edge-inference.tar.gz.sha256` give values for each new build. Peak resident memory, cold/warm inference latency, board storage usage, and 2 GB UNO Q fit are **PENDING physical measurement**. Capture them on the board with `/usr/bin/time -v ./bin/edge-validate ...` if that normal Debian utility is available, and repeat an inference for warm timing.

On 2026-10-23, the archive with that SHA-256 passed `edge-run --check`, produced the expected MNIST class 7, and passed `edge-validate` on a physical UNO Q. The device reported Debian 13.1, AArch64, kernel `6.16.7-g0dd6551ae96b`, and system glibc 2.41. The runner reported bundled glibc 2.42 and ONNX Runtime 1.24.4. An ONNX Runtime GPU discovery warning appeared, with no effect on CPU inference.
