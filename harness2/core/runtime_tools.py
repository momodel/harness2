"""Explicit, content-checked local tools for clean OpenCode runs.

Trusted local supply, not a binary signature/security boundary. Never discover
binaries on the operator's PATH or download them during a rollout.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .. import config


def root():
    return config.DEPENDENCIES / "tools" / "clean-rg"


def identity():
    directory = root()
    try:
        record = json.loads((directory / "manifest.json").read_text())
        binary = directory / "rg"
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        if (record.get("version") != 1 or record.get("sha256") != actual
                or not os.access(binary, os.X_OK) or not record.get("rg_version")):
            raise ValueError("binary/manifest mismatch")
        return {"version": 1, "sha256": actual, "rg_version": record["rg_version"]}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise config.ConfigError(
            "Clean ripgrep missing or changed; run python -m harness2.cli.ecom_tools --rg /absolute/path/to/rg"
        ) from exc


def install(source: Path):
    """Snapshot only the explicitly approved executable, never its whole PATH."""
    if not source.is_absolute() or not source.is_file():
        raise ValueError("--rg must name an existing absolute executable path")
    directory = root()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clean-rg-", dir=directory.parent) as temp:
        staged = Path(temp)
        binary = staged / "rg"
        shutil.copyfile(source, binary)
        binary.chmod(0o755)
        result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, check=True, timeout=10,
            env={"PATH": "/usr/bin:/bin", "HOME": str(staged)},
        )
        version = result.stdout.splitlines()[0]
        if not version.startswith("ripgrep "):
            raise ValueError("Supplied executable did not identify itself as ripgrep")
        record = {"version": 1, "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                  "rg_version": version}
        if directory.exists():
            if identity() == record:
                return record
            raise ValueError("Refusing to replace a different tool snapshot; move clean-rg aside explicitly")
        (staged / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        shutil.copytree(staged, directory)
    return identity()


def materialise(workspace: Path):
    record = identity()
    target = workspace / ".harness2-env" / "bin"
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(root() / "rg", target / "rg")
    (target / "rg").chmod(0o755)
    if hashlib.sha256((target / "rg").read_bytes()).hexdigest() != record["sha256"]:
        raise config.ConfigError("Clean ripgrep changed during materialisation")
    return target
