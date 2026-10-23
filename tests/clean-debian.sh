#!/bin/sh
# Run with Docker on an ARM64 Linux host: tests the transferred archive without Nix.
set -eu
archive=$(realpath "$1")
tar -xOzf "$archive" edge-inference/bin/edge-native | file - | grep -q 'ELF 64-bit.*ARM aarch64'
docker run --rm --platform linux/arm64 --network none \
  -v "$archive:/tmp/edge-inference.tar.gz:ro" debian:trixie-slim sh -eu -c '
    mkdir /tmp/test
    cd /tmp/test
    tar -xzf /tmp/edge-inference.tar.gz
    cd edge-inference
    test ! -e /nix/store
    ./bin/edge-run --check
    ./bin/edge-run --input validation/sample.json
    ./bin/edge-validate --input validation/sample.json --expected validation/expected.json
    cp -R . /tmp/corrupt
    printf x >> /tmp/corrupt/model/model.onnx
    if /tmp/corrupt/bin/edge-run --check; then exit 1; fi
    cp -R . /tmp/missing
    rm /tmp/missing/lib/libonnxruntime.so.1
    if /tmp/missing/bin/edge-run --check; then exit 1; fi
    printf "{\"Input3\":[1]}\n" > /tmp/bad-input.json
    if ./bin/edge-run --input /tmp/bad-input.json; then exit 1; fi
    printf "{\"outputs\":{\"Plus214_Output_0\":{\"argmax\":0}}}\n" > /tmp/bad-expected.json
    if ./bin/edge-validate --input validation/sample.json --expected /tmp/bad-expected.json; then exit 1; fi
    printf "{\"outputs\":{}}\n" > /tmp/empty-expected.json
    if ./bin/edge-validate --input validation/sample.json --expected /tmp/empty-expected.json; then exit 1; fi
  '
