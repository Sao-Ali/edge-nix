{
  description = "Example project that packages its own ONNX model with edge-nix";

  inputs.edgeNix.url = "github:Sao-Ali/edge-nix";
  inputs.nixpkgs.follows = "edgeNix/nixpkgs";

  outputs = { self, edgeNix, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
      forSystems = f: nixpkgs.lib.genAttrs systems
        (system: f (import nixpkgs { inherit system; }));
    in
    {
      packages = forSystems (pkgs: {
        default = edgeNix.lib.mkOnnxBundle pkgs {
          name = "consumer-model";
          modelFile = ./model.onnx;
          maxModelBytes = 4096;
        };
      });

      checks = forSystems (pkgs: {
        package = pkgs.runCommand "consumer-package-validation" { } ''
          ${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/edge-validate \
            @${./sample.json} @${./expected.json} > "$out"
        '';
      });
    };
}
