"""Copy the ELF dependency closure and make a deterministic UNO Q candidate archive."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elf(path):
    with path.open("rb") as source:
        return source.read(4) == b"\x7fELF"


def inspect(path):
    header = command("readelf", "-h", str(path))
    if "AArch64" not in header or "ELF64" not in header:
        raise RuntimeError(f"not an ELF64 AArch64 binary: {path}")
    needed = command("patchelf", "--print-needed", str(path)).splitlines()
    rpath = command("patchelf", "--print-rpath", str(path))
    if "/nix/store/" in rpath:
        raise RuntimeError(f"unrelocated RPATH: {path}: {rpath}")
    return needed


def dependencies(path):
    output = command("ldd", str(path))
    found = {}
    for line in output.splitlines():
        if "not found" in line:
            raise RuntimeError(f"unresolved library for {path}: {line}")
        match = re.search(r"^\s*(\S+)\s+=>\s+(/\S+)", line)
        if match:
            found[match.group(1)] = Path(match.group(2))
    return found


def main():
    root = Path(sys.argv[1])
    native = root / "bin/edge-native"
    lib = root / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    interpreter = Path(command("patchelf", "--print-interpreter", str(native)))
    if interpreter.name != "ld-linux-aarch64.so.1":
        raise RuntimeError(f"unexpected ELF interpreter: {interpreter}")
    shutil.copyfile(interpreter, lib / interpreter.name)
    os.chmod(lib / interpreter.name, 0o755)
    pending = [native]
    seen = set()
    origins = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        resolved = dependencies(path)
        for name in command("patchelf", "--print-needed", str(path)).splitlines():
            if name == interpreter.name:
                continue
            if name not in resolved:
                raise RuntimeError(f"cannot resolve DT_NEEDED {name} for {path}")
            source = resolved[name].resolve()
            target = lib / name
            if target.exists() and source == target.resolve():
                continue
            if not str(source).startswith("/nix/store/"):
                raise RuntimeError(f"unexpected non-Nix dependency: {source}")
            if target.exists():
                if origins[name] != sha256(source):
                    raise RuntimeError(f"conflicting SONAME {name}")
                continue
            shutil.copyfile(source, target)
            os.chmod(target, 0o755)
            origins[name] = sha256(source)
            pending.append(target)
    command("patchelf", "--set-interpreter", "/lib/ld-linux-aarch64.so.1", str(native))
    for path in sorted([native, *lib.iterdir()]):
        if path.name != interpreter.name:
            command("patchelf", "--set-rpath", "$ORIGIN/../lib" if path == native else "$ORIGIN", str(path))
        inspect(path)
        if path == native and command("patchelf", "--print-interpreter", str(path)) != "/lib/ld-linux-aarch64.so.1":
            raise RuntimeError("native runner has unexpected interpreter")
    # The shell entry points always call our bundled loader explicitly. The
    # ordinary PT_INTERP is patched as a safe fallback, not relied on.
    model = root / "model/model.onnx"
    prepared = json.loads((root / "prepared-manifest.json").read_text())
    config = json.loads((root / "config/inference.json").read_text())
    files = {}
    embedded = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in bundle: {path}")
        if path.is_file() and path.name not in ("manifest.json", "prepared-manifest.json"):
            relative = str(path.relative_to(root))
            files[relative] = sha256(path)
            count = path.read_bytes().count(b"/nix/store/")
            if count:
                if not elf(path):
                    raise RuntimeError(f"Nix store reference in non-ELF bundle file: {path}")
                embedded[relative] = count
    manifest = {
        "schemaVersion": 3,
        "artifactType": "inference-bundle",
        "name": config["name"],
        "target": "arduino-uno-q",
        "architecture": "aarch64-linux",
        "executionProvider": "CPUExecutionProvider",
        "quantization": config["quantization"],
        "onnxRuntimeVersion": config["onnxRuntimeVersion"],
        "onnxVersion": config["onnxVersion"],
        "nixpkgsRev": config["nixpkgsRev"],
        "recipeSourceSha256": config["recipeSourceSha256"],
        "runnerVersion": 1,
        "sourceModelSha256": prepared["sourceModelSha256"],
        "sourceModelBytes": prepared["sourceModelBytes"],
        "preparedModelSha256": sha256(model),
        "preparedModelBytes": model.stat().st_size,
        "inputs": prepared["inputs"],
        "outputs": prepared["outputs"],
        "externalAbi": {"os": "Debian 13 Linux", "architecture": "aarch64", "loader": "bundled glibc 2.42", "kernel": ">=6.1"},
        "files": files,
        "dependencies": sorted(str(path.relative_to(root)) for path in lib.iterdir()),
        "embeddedStoreReferenceCounts": embedded,
        "validation": {"input": "validation/sample.json", "expected": "validation/expected.json"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    (root / "prepared-manifest.json").unlink()


if __name__ == "__main__":
    main()
