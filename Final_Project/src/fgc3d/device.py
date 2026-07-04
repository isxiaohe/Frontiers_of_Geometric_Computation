"""Device selection helpers for local and server runs."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a user-facing device string into a torch device.

    ``auto`` intentionally prefers CUDA but falls back to CPU instead of MPS.
    The current voxel path relies on operations that are more predictable on
    CPU/CUDA, and the server target for this project is NVIDIA CUDA.
    """

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device!r}, but torch.cuda.is_available() is false")
    if resolved.type == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError(f"requested MPS device {device!r}, but MPS is not available")
    return resolved
