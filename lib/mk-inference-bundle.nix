{ pkgs, nixpkgsRev }:
{ name
, modelFile
, target ? "arduino-uno-q"
, backend ? "onnxruntime-cpu"
, quantization ? "none"
, sampleInput
, expectedOutput
, maxModelBytes ? null
}:
let
  lib = pkgs.lib;
  buildPython = pkgs.python3.withPackages (ps: [ ps.numpy ps.onnx ps.onnxruntime ps.sympy ]);
  config = pkgs.writeText "${name}-inference-config.json" (builtins.toJSON {
    inherit name quantization nixpkgsRev;
    onnxRuntimeVersion = pkgs.onnxruntime.version;
    onnxVersion = pkgs.python3Packages.onnx.version;
    recipeSourceSha256 = {
      runner = builtins.hashFile "sha256" ./inference-runner.cc;
      packager = builtins.hashFile "sha256" ./package-inference.py;
      bundle = builtins.hashFile "sha256" ./mk-inference-bundle.nix;
    };
  });
  prepared = pkgs.runCommand "${name}-prepared" { nativeBuildInputs = [ buildPython ]; } ''
    mkdir -p "$out"
    ${if quantization == "dynamic-int8" then ''
      cp ${modelFile} input.onnx
      python ${./quantize.py} input.onnx "$out/model.onnx"
    '' else ''
      cp ${modelFile} "$out/model.onnx"
    ''}
    ${lib.optionalString (maxModelBytes != null) ''
      test "$(wc -c < "$out/model.onnx")" -le ${toString maxModelBytes} || {
        echo 'edge-nix: prepared model exceeds maxModelBytes' >&2; exit 1;
      }
    ''}
    python ${./manifest.py} "$out/model.onnx" ${config} ${modelFile}
  '';
  runner = pkgs.stdenv.mkDerivation {
    pname = "edge-inference-runner";
    version = "1";
    dontUnpack = true;
    buildInputs = [ pkgs.onnxruntime pkgs.onnxruntime.dev pkgs.nlohmann_json pkgs.openssl ];
    buildPhase = ''
      runHook preBuild
      c++ -std=c++17 -O2 -I${pkgs.onnxruntime.dev}/include \
        -I${pkgs.nlohmann_json}/include -I${pkgs.openssl.dev}/include \
        ${./inference-runner.cc} -L${pkgs.onnxruntime}/lib -lonnxruntime \
        -L${pkgs.openssl.out}/lib -lcrypto -o edge-native
      runHook postBuild
    '';
    installPhase = ''
      mkdir -p "$out/bin"
      cp edge-native "$out/bin/edge-native"
    '';
  };
in
assert lib.assertMsg (target == "arduino-uno-q") "edge-nix: only target arduino-uno-q is implemented";
assert lib.assertMsg (backend == "onnxruntime-cpu") "edge-nix: only backend onnxruntime-cpu is implemented";
assert lib.assertMsg (builtins.elem quantization [ "none" "dynamic-int8" ]) "edge-nix: unsupported quantization";
assert lib.assertMsg (maxModelBytes == null || (builtins.isInt maxModelBytes && maxModelBytes > 0))
  "edge-nix: maxModelBytes must be a positive integer or null";
assert lib.assertMsg (pkgs.stdenv.hostPlatform.system == "aarch64-linux" && pkgs.stdenv.buildPlatform.system == "aarch64-linux")
  "edge-nix: UNO Q inference bundles require a native aarch64-linux builder";
pkgs.runCommand name {
  nativeBuildInputs = [ pkgs.python3 pkgs.patchelf pkgs.binutils pkgs.gnutar pkgs.gzip pkgs.file ];
} ''
  mkdir -p "$out" edge-inference/{bin,model,lib,config,validation,LICENSES}
  cp ${runner}/bin/edge-native edge-inference/bin/edge-native
  cp ${prepared}/model.onnx edge-inference/model/model.onnx
  cp ${prepared}/manifest.json edge-inference/prepared-manifest.json
  cp ${config} edge-inference/config/inference.json
  cp ${sampleInput} edge-inference/validation/sample.json
  cp ${expectedOutput} edge-inference/validation/expected.json
  cp ${../LICENSE} edge-inference/LICENSES/edge-nix-MIT.txt
  cp ${../licenses}/* edge-inference/LICENSES/
  cat > edge-inference/LICENSES/DEPENDENCIES.txt <<'EOF'
  ONNX Runtime 1.24.4: MIT; onnxruntime-MIT.txt
  nlohmann/json 3.12.0: MIT; nlohmann-json-MIT.txt
  OpenSSL 3.6.5: Apache-2.0; openssl-Apache-2.0.txt
  glibc 2.42: LGPL-2.1-or-later; glibc-LGPL-2.1.txt
  GCC runtime libraries: GPL-3.0-or-later with GCC Runtime Library Exception 3.1;
    gcc-GPL-3.0.txt and gcc-runtime-exception-3.1.txt
  Abseil 20260107.1: Apache-2.0; abseil-Apache-2.0.txt
  Protocol Buffers 34.1: BSD-3-Clause; protobuf-BSD-3-Clause.txt
  Binary dependency names and SHA-256 hashes are in manifest.json.
  EOF
  cat > edge-inference/bin/edge-run <<'EOF'
  #!/bin/sh
  set -eu
  root=$(CDPATH= cd -- "''${0%/*}/.." && pwd -P)
  exec "$root/lib/ld-linux-aarch64.so.1" --library-path "$root/lib" "$root/bin/edge-native" "$root" "$@"
  EOF
  cp edge-inference/bin/edge-run edge-inference/bin/edge-validate
  chmod 755 edge-inference/bin/*
  python ${./package-inference.py} edge-inference
  # The Nix build environment need not be Debian; clean-userland CI exercises --check.
  edge-inference/bin/edge-run --input edge-inference/validation/sample.json > check.json
  edge-inference/bin/edge-validate --input edge-inference/validation/sample.json \
    --expected edge-inference/validation/expected.json > validate.json
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -cf - edge-inference | gzip -n > "$out/edge-inference.tar.gz"
  cp check.json validate.json "$out/"
  file edge-inference/bin/edge-native > "$out/elf.txt"
  wc -c "$out/edge-inference.tar.gz" > "$out/size.txt"
  (cd "$out" && sha256sum edge-inference.tar.gz > edge-inference.tar.gz.sha256)
''
