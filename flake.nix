{
  description = "Build model artifacts for edge inference with Nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
      linuxSystems = [ "x86_64-linux" "aarch64-linux" ];
      forSystems = f: nixpkgs.lib.genAttrs systems
        (system: f (import nixpkgs { inherit system; }));
      forLinux = f: nixpkgs.lib.genAttrs linuxSystems
        (system: f (import nixpkgs { inherit system; }));
    in
    {
      lib.mkOnnxBundle = pkgs: import ./lib/mk-onnx-bundle.nix { inherit pkgs; };

      packages = forSystems (pkgs:
        let
          mkOnnxBundle = self.lib.mkOnnxBundle pkgs;
          model = import ./examples/mnist/model.nix { inherit pkgs; };
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
          default = self.packages.${pkgs.stdenv.hostPlatform.system}.demo-int8;
        });

      checks = forLinux (pkgs: {
        vm = import ./tests/vm.nix {
          inherit pkgs;
          fp32 = self.packages.${pkgs.stdenv.hostPlatform.system}.demo-fp32;
          int8 = self.packages.${pkgs.stdenv.hostPlatform.system}.demo-int8;
        };
      });
    };
}
