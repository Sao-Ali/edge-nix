{
  description = "Build model artifacts for edge inference with Nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
      forSystems = f: nixpkgs.lib.genAttrs systems
        (system: f (import nixpkgs { inherit system; }));
    in
    {
      lib.mkOnnxBundle = pkgs: import ./lib/mk-onnx-bundle.nix { inherit pkgs; };
      lib.mkInferenceBundle = pkgs: import ./lib/mk-inference-bundle.nix {
        inherit pkgs;
        nixpkgsRev = nixpkgs.rev or "unknown";
      };
      lib.mkEdgeSystem = pkgs: import ./lib/mk-edge-system.nix {
        inherit pkgs;
        mkInferenceBundle = self.lib.mkInferenceBundle pkgs;
      };

      packages = forSystems (pkgs:
        let
          mkOnnxBundle = self.lib.mkOnnxBundle pkgs;
          model = import ./examples/mnist/model.nix { inherit pkgs; };
          integerModel = pkgs.runCommand "int64-identity.onnx" {
            nativeBuildInputs = [ (pkgs.python3.withPackages (ps: [ ps.onnx ])) ];
          } ''
            python ${./examples/int64-identity/generate.py} "$out"
          '';
        in
        {
          demo-fp32 = mkOnnxBundle {
            name = "edge-nix-demo-fp32";
            modelFile = model;
          };
          demo-int8 = mkOnnxBundle {
            name = "edge-nix-demo-int8";
            modelFile = model;
            quantization = "dynamic-int8";
            maxModelBytes = 1048576;
          };
          demo-int64 = mkOnnxBundle {
            name = "edge-nix-demo-int64";
            modelFile = integerModel;
          };
          default = self.packages.${pkgs.stdenv.hostPlatform.system}.demo-int8;
        } // pkgs.lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "aarch64-linux") {
          uno-q-demo = self.lib.mkInferenceBundle pkgs {
            name = "mnist-uno-q";
            modelFile = model;
            sampleInput = ./examples/mnist/sample.json;
            expectedOutput = ./examples/mnist/expected.json;
          };
          uno-q-demo-int8 = self.lib.mkInferenceBundle pkgs {
            name = "mnist-uno-q-int8";
            modelFile = model;
            quantization = "dynamic-int8";
            maxModelBytes = 1048576;
            sampleInput = ./examples/mnist/sample.json;
            expectedOutput = ./examples/mnist/expected.json;
          };
        });

      checks = forSystems (pkgs:
        let
          packages = self.packages.${pkgs.stdenv.hostPlatform.system};
          model = import ./examples/mnist/model.nix { inherit pkgs; };
          configuredUnoQ = self.lib.mkEdgeSystem pkgs {
            name = "mnist-uno-q";
            inherit model;
            target = { device = "arduino-uno-q"; accelerator = "cpu"; };
            runtime.backend = "onnxruntime";
            validation = {
              sampleInput = ./examples/mnist/sample.json;
              expectedOutput = ./examples/mnist/expected.json;
            };
          };
          configuredUnoQInt8 = self.lib.mkEdgeSystem pkgs {
            name = "mnist-uno-q-int8";
            inherit model;
            target = { device = "arduino-uno-q"; accelerator = "cpu"; };
            runtime.backend = "onnxruntime";
            quantization = { enable = true; type = "dynamic-int8"; };
            maxModelBytes = 1048576;
            validation = {
              sampleInput = ./examples/mnist/sample.json;
              expectedOutput = ./examples/mnist/expected.json;
            };
          };
          mkValidation = bundle: sample: expected: extraArgs:
            pkgs.runCommand "${bundle.name}-validation" { } ''
            ${bundle}/bin/edge-validate \
              @${sample} \
              @${expected} ${extraArgs} > "$out"
          '';
        in
        {
          demo-fp32 = mkValidation packages.demo-fp32
            ./examples/mnist/sample.json ./examples/mnist/expected.json "";
          demo-int8 = mkValidation packages.demo-int8
            ./examples/mnist/sample.json ./examples/mnist/expected.json
            "--reference-archive ${packages.demo-fp32}/edge-package.tar.gz --atol 0.25 --rtol 0.01";
          demo-int64 = mkValidation packages.demo-int64
            ./examples/int64-identity/sample.json ./examples/int64-identity/expected.json "";
        } // pkgs.lib.optionalAttrs (pkgs.stdenv.hostPlatform.system == "aarch64-linux") {
          uno-q-config-equivalence =
            assert pkgs.lib.assertMsg (configuredUnoQ.drvPath == packages.uno-q-demo.drvPath)
              "edge-nix: configuration API changed the UNO Q bundle derivation";
            assert pkgs.lib.assertMsg (configuredUnoQInt8.drvPath == packages.uno-q-demo-int8.drvPath)
              "edge-nix: configuration API changed the INT8 UNO Q bundle derivation";
            pkgs.runCommand "uno-q-config-equivalence" { } ''
              cmp ${configuredUnoQ}/edge-inference.tar.gz ${packages.uno-q-demo}/edge-inference.tar.gz
              cmp ${configuredUnoQ}/check.json ${packages.uno-q-demo}/check.json
              cmp ${configuredUnoQ}/validate.json ${packages.uno-q-demo}/validate.json
              cmp ${configuredUnoQInt8}/edge-inference.tar.gz ${packages.uno-q-demo-int8}/edge-inference.tar.gz
              cmp ${configuredUnoQInt8}/check.json ${packages.uno-q-demo-int8}/check.json
              cmp ${configuredUnoQInt8}/validate.json ${packages.uno-q-demo-int8}/validate.json
              cp ${configuredUnoQ}/validate.json "$out"
            '';
          uno-q-demo = pkgs.runCommand "uno-q-demo-check" { } ''
            cp ${packages.uno-q-demo}/validate.json "$out"
          '';
          uno-q-demo-int8 = pkgs.runCommand "uno-q-demo-int8-check" { } ''
            cp ${packages.uno-q-demo-int8}/validate.json "$out"
          '';
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          vm = import ./tests/vm.nix {
            inherit pkgs;
            fp32 = packages.demo-fp32;
            int8 = packages.demo-int8;
          };
        });
    };
}
