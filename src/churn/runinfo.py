"""Facts logged with every MLflow run, kept free of heavy imports."""

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash of a data file, logged so a run can be tied to the exact data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    """Current commit, logged with every MLflow run."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
