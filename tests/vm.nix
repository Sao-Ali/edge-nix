{ pkgs, fp32, int8 }:
pkgs.testers.runNixOSTest {
  name = "edge-nix-onnx";
  nodes.machine = { ... }: {
    environment.systemPackages = [ fp32 int8 ];
  };
  testScript = ''
    machine.start()
    sample = "@${../examples/mnist/sample.json}"
    expected = "@${../examples/mnist/expected.json}"
    for bundle in ["${fp32}", "${int8}"]:
        machine.succeed(bundle + "/bin/edge-validate " + sample + " " + expected)
  '';
}
