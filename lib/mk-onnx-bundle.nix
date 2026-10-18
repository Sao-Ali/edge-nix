{ pkgs }:
{ name
, modelFile
, quantization ? "none"
, maxModelBytes ? null
}:
let
  lib = pkgs.lib;
  supportedQuantization = [ "none" "dynamic-int8" ];
  runtimePython = pkgs.python3.withPackages (ps: [ ps.numpy ps.onnxruntime ]);
  buildPython = pkgs.python3.withPackages (ps: [ ps.numpy ps.onnx ps.onnxruntime ps.sympy ]);
  manifestConfig = pkgs.writeText "${name}-manifest-config.json" (builtins.toJSON {
    inherit name quantization maxModelBytes;
    executionProvider = "CPUExecutionProvider";
    format = "onnx";
    runtime = "onnxruntime";
    runtimeVersion = pkgs.python3Packages.onnxruntime.version;
    pythonVersion = pkgs.python3.version;
    numpyVersion = pkgs.python3Packages.numpy.version;
    targetSystem = pkgs.stdenv.hostPlatform.system;
  });
  preparedModel = pkgs.runCommand "${name}-model" {
    nativeBuildInputs = [ buildPython ];
  } ''
    mkdir -p "$out"
    ${if quantization == "dynamic-int8" then ''
      cp "${modelFile}" ./input.onnx
      python ${./quantize.py} ./input.onnx "$out/model.onnx"
    '' else ''
      cp "${modelFile}" "$out/model.onnx"
    ''}
    ${lib.optionalString (maxModelBytes != null) ''
      model_bytes=$(wc -c < "$out/model.onnx")
      if [ "$model_bytes" -gt ${toString maxModelBytes} ]; then
        echo "edge-nix: model exceeds ${toString maxModelBytes} byte limit" >&2
        exit 1
      fi
    ''}
    python ${./manifest.py} "$out/model.onnx" ${manifestConfig} "${modelFile}"
  '';
in
assert lib.assertMsg (builtins.elem quantization supportedQuantization)
  "edge-nix: quantization must be one of ${lib.concatStringsSep ", " supportedQuantization}";
assert lib.assertMsg (maxModelBytes == null || (builtins.isInt maxModelBytes && maxModelBytes > 0))
  "edge-nix: maxModelBytes must be a positive integer or null";
pkgs.runCommand name { nativeBuildInputs = [ pkgs.gnutar pkgs.gzip ]; } ''
  mkdir -p "$out/bin"
  mkdir -p package
  cp ${preparedModel}/model.onnx package/model.onnx
  cp ${preparedModel}/manifest.json package/manifest.json
  if grep -a -q '/nix/store/' package/model.onnx package/manifest.json; then
    echo "edge-nix: package contains a host Nix store reference" >&2
    exit 1
  fi
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -C package -cf - manifest.json model.onnx | gzip -n > "$out/edge-package.tar.gz"
  ln -s ${preparedModel}/model.onnx "$out/model.onnx"
  ln -s ${preparedModel}/manifest.json "$out/manifest.json"
  cat > "$out/bin/edge-run" <<'EOF'
#!${pkgs.runtimeShell}
exec ${runtimePython}/bin/python ${./run.py} ${preparedModel}/model.onnx "$@"
EOF
  cat > "$out/bin/edge-validate" <<'EOF'
#!${pkgs.runtimeShell}
export PYTHONPATH=${./.}
exec ${runtimePython}/bin/python ${./validate.py} "$(dirname "$0")/../edge-package.tar.gz" "$@"
EOF
  chmod +x "$out/bin/edge-run" "$out/bin/edge-validate"
''
