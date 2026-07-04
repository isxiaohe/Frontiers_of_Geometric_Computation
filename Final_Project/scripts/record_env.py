"""Record a compact environment snapshot for migration and debugging."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "mps_available": False,
            "device_count": 0,
            "devices": [],
            "import_error": "torch is not installed",
        }

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]

    return {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": devices,
    }


def collect_environment() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "huggingface_hub", "torch"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    torch_info = _torch_info()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "torch": torch_info,
        "project": {
            "requires_python": ">=3.11,<3.13",
            "base_requirements": "env/requirements-base.txt",
            "cpu_requirements": "env/requirements-cpu.txt",
            "cuda_requirements_example": "env/requirements-cuda-cu128.txt",
            "test_command": "PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v",
            "recommended_device_arg": "cuda:0" if torch_info["cuda_available"] else "cpu",
        },
        "note": "Local verification snapshot only. Recreate .venv on CUDA machines and refresh this file.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    data = collect_environment()
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
