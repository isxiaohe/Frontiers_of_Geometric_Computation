"""Lightweight point-cloud generation metrics."""

from __future__ import annotations

import torch


def chamfer_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Symmetric squared Chamfer distance between two point clouds."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[-1] != 3 or second.shape[-1] != 3:
        raise ValueError("point clouds must have shape (points, 3)")
    distances = torch.cdist(first, second, p=2).square()
    return distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()


def pairwise_chamfer_distances(samples: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return pairwise Chamfer distances with shape (num_samples, num_reference)."""
    if samples.ndim != 3 or reference.ndim != 3 or samples.shape[-1] != 3 or reference.shape[-1] != 3:
        raise ValueError("point-cloud batches must have shape (batch, points, 3)")
    distances = torch.empty(samples.shape[0], reference.shape[0], dtype=samples.dtype, device=samples.device)
    for sample_index in range(samples.shape[0]):
        for reference_index in range(reference.shape[0]):
            distances[sample_index, reference_index] = chamfer_distance(
                samples[sample_index],
                reference[reference_index].to(samples.device),
            )
    return distances


def point_cloud_distribution_metrics(samples: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compute small-set MMD-CD and Coverage-CD metrics."""
    if samples.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("samples and reference must be non-empty")
    distances = pairwise_chamfer_distances(samples, reference.to(samples.device))
    nearest_sample_for_reference = distances.min(dim=0)
    mmd_cd = nearest_sample_for_reference.values.mean()
    coverage = nearest_sample_for_reference.indices.unique().numel() / float(samples.shape[0])
    return {
        "mmd_cd": float(mmd_cd.detach().cpu()),
        "coverage": float(coverage),
    }
