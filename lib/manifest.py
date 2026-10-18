"""Write a manifest for the model that was actually built."""

import hashlib
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


def tensor_info(value):
    return {"name": value.name, "type": value.type, "shape": value.shape}


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def has_external_data(message):
    if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
        return True
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        children = value if field.is_repeated else [value]
        if any(has_external_data(child) for child in children):
            return True
    return False


def main():
    model_path = Path(sys.argv[1])
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    source_path = Path(sys.argv[3])
    model = onnx.load_model(str(model_path), load_external_data=False)
    if has_external_data(model):
        raise SystemExit("edge-nix: ONNX external data files are not supported")
    onnx.checker.check_model(str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_size, model_hash = file_info(model_path)
    source_size, source_hash = file_info(source_path)
    manifest = {
        **config,
        "manifestVersion": 2,
        "modelFile": "model.onnx",
        "modelBytes": model_size,
        "modelSha256": model_hash,
        "sourceModelBytes": source_size,
        "sourceModelSha256": source_hash,
        "inputs": [tensor_info(value) for value in session.get_inputs()],
        "outputs": [tensor_info(value) for value in session.get_outputs()],
    }
    (model_path.parent / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
