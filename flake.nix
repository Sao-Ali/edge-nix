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
        });

      checks = forSystems (pkgs:
        let
          packages = self.packages.${pkgs.stdenv.hostPlatform.system};
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
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          vm = import ./tests/vm.nix {
            inherit pkgs;
            fp32 = packages.demo-fp32;
            int8 = packages.demo-int8;
          };
        });
    };
}
