# Environment Reproduction

This project should be migrated by recreating `.venv`, not by copying `.venv`
from another machine.

## CPU / macOS / generic setup

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-cpu.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## NVIDIA CUDA setup

Pick the PyTorch CUDA wheel index that matches the target machine. Example for
CUDA 12.8 wheels:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-base.txt
uv pip install -r env/requirements-cuda-cu128.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python scripts/record_env.py --output env/current-env.json
```

For another CUDA version, replace `cu128` with the index recommended by the
PyTorch install selector.

## Current Mac verification snapshot

`current-env.json` records the environment used for local verification on this
machine. It is an audit record, not a portable lockfile. The key portability
contract is:

- Python: `>=3.11,<3.13`
- Runtime dependencies: `numpy>=1.26,<3`, `huggingface_hub>=0.24,<1`,
  `torch>=2.3,<3`
- Server device selection: pass `--device auto` or `--device cuda:0` to
  training/generation scripts.
- Tests: standard-library `unittest`
- Project code: use `PYTHONPATH=src` unless the package is installed editable

## Refresh the snapshot

After recreating the environment on a new machine:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_env.py --output env/current-env.json
```
