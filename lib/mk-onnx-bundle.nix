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
  quantizerPython = pkgs.python3.withPackages (ps: [ ps.numpy ps.onnx ps.onnxruntime ps.sympy ]);
  preparedModel = pkgs.runCommand "${name}-model" {
    nativeBuildInputs = lib.optional (quantization == "dynamic-int8") quantizerPython;
  } ''
    mkdir -p "$out"
    ${if quantization == "dynamic-int8" then ''
      cp ${lib.escapeShellArg (toString modelFile)} ./input.onnx
      python ${./quantize.py} ./input.onnx "$out/model.onnx"
    '' else ''
      cp ${lib.escapeShellArg (toString modelFile)} "$out/model.onnx"
    ''}
    ${lib.optionalString (maxModelBytes != null) ''
      model_bytes=$(wc -c < "$out/model.onnx")
      if [ "$model_bytes" -gt ${toString maxModelBytes} ]; then
        echo "edge-nix: model exceeds ${toString maxModelBytes} byte limit" >&2
        exit 1
      fi
    ''}
  '';
  manifest = pkgs.writeText "${name}-manifest.json" (builtins.toJSON {
    inherit name quantization;
    inherit maxModelBytes;
    executionProvider = "CPUExecutionProvider";
    format = "onnx";
    model = toString modelFile;
    runtime = "onnxruntime";
    runtimeVersion = pkgs.python3Packages.onnxruntime.version;
    targetSystem = pkgs.stdenv.hostPlatform.system;
  });
in
assert lib.assertMsg (builtins.elem quantization supportedQuantization)
  "edge-nix: quantization must be one of ${lib.concatStringsSep ", " supportedQuantization}";
assert lib.assertMsg (maxModelBytes == null || (builtins.isInt maxModelBytes && maxModelBytes > 0))
  "edge-nix: maxModelBytes must be a positive integer or null";
pkgs.runCommand name { } ''
  mkdir -p "$out/bin"
  ln -s ${preparedModel}/model.onnx "$out/model.onnx"
  ln -s ${manifest} "$out/manifest.json"
  cat > "$out/bin/edge-run" <<'EOF'
#!${pkgs.runtimeShell}
exec ${runtimePython}/bin/python ${./run.py} ${preparedModel}/model.onnx "$@"
EOF
  chmod +x "$out/bin/edge-run"
''
