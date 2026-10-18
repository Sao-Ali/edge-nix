# Use edge-nix from another project

This directory is a separate flake with its own `model.onnx`, input fixture, and expected output. Its `flake.nix` adds `edge-nix` as a dependency and builds a portable archive.

From this directory, run:

```sh
nix build
tar -tzf result/edge-package.tar.gz
./result/bin/edge-validate @sample.json @expected.json
nix flake check
```

The archive contains only `manifest.json` and `model.onnx`. The `bin` commands in `result` run on the build host and are not part of the archive. Replace `model.onnx`, `sample.json`, and `expected.json` with your own files, then adjust `maxModelBytes` in `flake.nix`.

To test local changes to `edge-nix` before they reach GitHub, add `--override-input edgeNix path:../.. --no-write-lock-file` to a `nix build` or `nix flake check` command.
