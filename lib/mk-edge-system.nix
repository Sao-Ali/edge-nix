{ pkgs, mkInferenceBundle }:
config:
let
  lib = pkgs.lib;
  unexpected = attrs: allowed: lib.subtractLists allowed (builtins.attrNames attrs);
  validFields = scope: attrs: allowed:
    lib.assertMsg (unexpected attrs allowed == [])
      "edge-nix: unsupported ${scope} fields: ${lib.concatStringsSep ", " (unexpected attrs allowed)}";
  target = config.target or { };
  runtime = config.runtime or { };
  quantization = config.quantization or { };
  validation = config.validation or { };
in
assert lib.assertMsg (builtins.isAttrs config) "edge-nix: configuration must be an attribute set";
assert lib.assertMsg (builtins.all builtins.isAttrs [ target runtime quantization validation ])
  "edge-nix: target, runtime, quantization, and validation must be attribute sets";
assert validFields "configuration" config [ "name" "model" "target" "runtime" "quantization" "validation" "maxModelBytes" ];
assert validFields "target" target [ "device" "accelerator" ];
assert validFields "runtime" runtime [ "backend" ];
assert validFields "quantization" quantization [ "enable" "type" ];
assert validFields "validation" validation [ "sampleInput" "expectedOutput" ];
assert lib.assertMsg (builtins.isString (config.name or null) && config.name != "")
  "edge-nix: name must be a nonempty string";
assert lib.assertMsg (config ? model) "edge-nix: model is required";
assert lib.assertMsg ((target.device or null) == "arduino-uno-q")
  "edge-nix: only target.device = \"arduino-uno-q\" is supported";
assert lib.assertMsg ((target.accelerator or null) == "cpu")
  "edge-nix: only target.accelerator = \"cpu\" is supported";
assert lib.assertMsg ((runtime.backend or null) == "onnxruntime")
  "edge-nix: only runtime.backend = \"onnxruntime\" is supported";
assert lib.assertMsg (builtins.isBool (quantization.enable or false))
  "edge-nix: quantization.enable must be a boolean";
assert lib.assertMsg ((quantization.type or "dynamic-int8") == "dynamic-int8")
  "edge-nix: only quantization.type = \"dynamic-int8\" is supported";
assert lib.assertMsg (validation ? sampleInput && validation ? expectedOutput)
  "edge-nix: validation.sampleInput and validation.expectedOutput are required";
mkInferenceBundle {
  inherit (config) name;
  modelFile = config.model;
  target = "arduino-uno-q";
  backend = "onnxruntime-cpu";
  quantization = if quantization.enable or false then "dynamic-int8" else "none";
  sampleInput = validation.sampleInput;
  expectedOutput = validation.expectedOutput;
  maxModelBytes = config.maxModelBytes or null;
}
