{ pkgs, fp32, int8 }:
pkgs.testers.runNixOSTest {
  name = "edge-nix-onnx";
  nodes.machine = { ... }: {
    environment.systemPackages = [ fp32 int8 ];
  };
  testScript = ''
    import json

    machine.start()
    sample = "@${../examples/mnist/sample.json}"
    for runner in ["${fp32}/bin/edge-run", "${int8}/bin/edge-run"]:
        result = json.loads(machine.succeed(runner + " " + sample))
        scores = result["Plus214_Output_0"][0]
        assert max(range(10), key=lambda index: scores[index]) == 7, result
  '';
}
