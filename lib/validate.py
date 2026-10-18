"""Check a built bundle against input and output fixtures."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from run import load_argument, run_model


PACKAGE_FILES = {"model.onnx", "manifest.json"}
STORE_MARKER = b"/nix/store/"


def unpack_archive(archive_path, directory):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != PACKAGE_FILES or len(members) != 2:
            raise ValueError("archive must contain only model.onnx and manifest.json")
        for member in members:
            if not member.isfile():
                raise ValueError(f"archive member is not a regular file: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            destination = directory / member.name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            if contains_store_reference(destination):
                raise ValueError(f"archive member contains a Nix store reference: {member.name}")
    return directory / "model.onnx", directory / "manifest.json"


def contains_store_reference(path):
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            if STORE_MARKER in data:
                return True
            tail = data[-(len(STORE_MARKER) - 1):]
    return False


def check_manifest(model_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("manifestVersion") != 2:
        raise ValueError("unsupported manifest version")
    if manifest.get("modelFile") != "model.onnx":
        raise ValueError("manifest modelFile must be model.onnx")
    model_file = Path(model_path)
    model_size = model_file.stat().st_size
    if model_size != manifest["modelBytes"]:
        raise ValueError("model size differs from manifest")
    digest = hashlib.sha256()
    with model_file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["modelSha256"]:
        raise ValueError("model hash differs from manifest")
    limit = manifest.get("maxModelBytes")
    if limit is not None and model_size > limit:
        raise ValueError("model exceeds maxModelBytes")
    return manifest


def check_outputs(actual, expected):
    if not isinstance(expected, dict):
        raise ValueError("expectations must be a JSON object")
    specifications = expected.get("outputs")
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError("expectations must contain a nonempty outputs object")
    checks = []
    for name, specification in specifications.items():
        if name not in actual:
            raise ValueError(f"missing output: {name}")
        if not isinstance(specification, dict) or not specification:
            raise ValueError(f"output {name}: expected a nonempty object")
        value = np.asarray(actual[name])
        if "shape" in specification:
            if list(value.shape) != specification["shape"]:
                raise ValueError(f"output {name}: shape {list(value.shape)} != {specification['shape']}")
            checks.append({"output": name, "check": "shape", "actual": list(value.shape)})
        if "argmax" in specification:
            index = int(np.argmax(value))
            if index != specification["argmax"]:
                raise ValueError(f"output {name}: argmax {index} != {specification['argmax']}")
            checks.append({"output": name, "check": "argmax", "actual": index})
        if "values" in specification:
            reference = np.asarray(specification["values"])
            tolerances = "atol" in specification or "rtol" in specification
            matches = value.shape == reference.shape and (
                np.allclose(
                    value, reference,
                    atol=specification.get("atol", 0),
                    rtol=specification.get("rtol", 0),
                ) if tolerances else np.array_equal(value, reference)
            )
            if not matches:
                raise ValueError(f"output {name}: values differ from expectation")
            checks.append({"output": name, "check": "values"})
        if not any(key in specification for key in ("shape", "argmax", "values")):
            raise ValueError(f"output {name}: no supported checks")
    return checks


def check_reference(actual, reference, atol, rtol):
    if actual.keys() != reference.keys():
        raise ValueError("model and reference have different output names")
    checks = []
    for name, values in actual.items():
        value = np.asarray(values)
        baseline = np.asarray(reference[name])
        if value.shape != baseline.shape:
            raise ValueError(f"output {name}: shape differs from reference")
        max_abs_diff = float(np.max(np.abs(value - baseline)))
        if not np.allclose(value, baseline, atol=atol, rtol=rtol):
            raise ValueError(f"output {name}: differs from reference (max absolute difference {max_abs_diff})")
        checks.append({"output": name, "check": "reference", "maxAbsDiff": max_abs_diff})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="edge-package.tar.gz")
    parser.add_argument("inputs", help="JSON object or @file.json")
    parser.add_argument("expectations", help="JSON object or @file.json")
    parser.add_argument("--reference-archive")
    parser.add_argument("--reference-model")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.reference_archive and args.reference_model:
        raise ValueError("choose one reference archive or reference model")
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        model_path, manifest_path = unpack_archive(args.archive, directory)
        manifest = check_manifest(model_path, manifest_path)
        supplied = load_argument(args.inputs)
        actual = run_model(str(model_path), supplied)
        checks = check_outputs(actual, load_argument(args.expectations))
        reference_model = args.reference_model
        if args.reference_archive:
            reference_directory = directory / "reference"
            reference_directory.mkdir()
            reference_model, reference_manifest = unpack_archive(
                args.reference_archive, reference_directory
            )
            check_manifest(reference_model, reference_manifest)
        if reference_model:
            reference = run_model(str(reference_model), supplied)
            checks.extend(check_reference(actual, reference, args.atol, args.rtol))
    print(json.dumps({"passed": True, "modelSha256": manifest["modelSha256"], "checks": checks}))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, tarfile.TarError, TypeError, ValueError) as error:
        raise SystemExit(f"edge-validate: {error}") from error
