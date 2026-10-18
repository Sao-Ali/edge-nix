"""Check a built bundle against input and output fixtures."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from run import load_argument, run_model


def check_manifest(model_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("manifestVersion") != 1:
        raise ValueError("unsupported manifest version")
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
    parser.add_argument("model_path")
    parser.add_argument("manifest_path")
    parser.add_argument("inputs", help="JSON object or @file.json")
    parser.add_argument("expectations", help="JSON object or @file.json")
    parser.add_argument("--reference-model")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    manifest = check_manifest(args.model_path, args.manifest_path)
    supplied = load_argument(args.inputs)
    actual = run_model(args.model_path, supplied)
    checks = check_outputs(actual, load_argument(args.expectations))
    if args.reference_model:
        reference = run_model(args.reference_model, supplied)
        checks.extend(check_reference(actual, reference, args.atol, args.rtol))
    print(json.dumps({"passed": True, "modelSha256": manifest["modelSha256"], "checks": checks}))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"edge-validate: {error}") from error
