"""Write a manifest for the model that was actually built."""

import hashlib
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


def tensor_info(value):
    return {"name": value.name, "type": value.type, "shape": value.shape}


def main():
    model_path = Path(sys.argv[1])
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    onnx.checker.check_model(str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        **config,
        "manifestVersion": 1,
        "modelFile": "model.onnx",
        "modelBytes": model_path.stat().st_size,
        "modelSha256": digest.hexdigest(),
        "inputs": [tensor_info(value) for value in session.get_inputs()],
        "outputs": [tensor_info(value) for value in session.get_outputs()],
    }
    (model_path.parent / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
