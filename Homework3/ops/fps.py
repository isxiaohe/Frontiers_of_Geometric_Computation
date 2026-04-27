"""Furthest Point Sampling — auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Furthest Point Sampling (FPS): greedily select `npoint` points that are
    maximally spread out in 3D space.

    Given N input points, FPS produces a subset of `npoint` indices such that
    each newly selected point is the one farthest (in Euclidean distance) from
    all previously selected points. This gives a roughly uniform spatial
    coverage — critical for the hierarchical structure of PointNet++.

    Args:
        xyz:    (B, N, 3) float tensor.
                B = batch size, N = number of input points, 3 = (x, y, z).
                Must be on CPU or GPU (matches CUDA_AVAILABLE).
        npoint: int, number of points to sample. Must be <= N.

    Returns:
        indices: (B, npoint) int64 tensor.
                 Each row contains npoint distinct indices in [0, N).
                 indices[b, 0] is the starting point (random).
                 indices[b, i] is the i-th sampled point for batch b.

    Example:
        >>> xyz = torch.rand(2, 1024, 3)          # 2 clouds, 1024 points each
        >>> idx = furthest_point_sample(xyz, 256)  # → shape (2, 256)
        >>> sampled_xyz = xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        >>> # sampled_xyz: (2, 256, 3) — the 256 FPS-selected points
    """
    if CUDA_AVAILABLE:
        from cuda_ops.fps import furthest_point_sample_cuda
        return furthest_point_sample_cuda(xyz, npoint)
    return _fps_pytorch(xyz, npoint)


def _fps_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Pure PyTorch FPS fallback.

    Algorithm (for each batch element, vectorized across B):
      1. Initialize:
         - distances = (B, N) filled with +inf  — tracks min dist to any selected pt
         - indices   = (B, npoint) zeros         — output array
         - farthest  = random int in [0, N)       — starting point index
      2. Set indices[:, 0] = farthest
      3. For i = 1, 2, ..., npoint-1:
         a. Get the last selected point's coordinates:
              centroid = xyz[torch.arange(B), farthest]   → (B, 3)
         b. Compute squared distance from centroid to ALL points:
              new_dist = ((xyz - centroid.unsqueeze(1)) ** 2).sum(dim=-1)  → (B, N)
         c. Update the running minimum:
              distances = torch.minimum(distances, new_dist)
         d. The next farthest point is the argmax of distances:
              farthest = distances.argmax(dim=-1)  → (B,)
         e. Store: indices[:, i] = farthest
      4. Return indices

    Complexity: O(npoint * N) — sequential outer loop, but the inner ops are
    batch-vectorized.

    Hints:
      - Use torch.arange(B, device=xyz.device) for batch-aware indexing
      - centroid.unsqueeze(1) broadcasts (B, 3) → (B, 1, 3) against (B, N, 3)
      - This loop is inherently sequential — can't be parallelized easily
    """
    # TODO: Implement FPS
    distances = torch.full((xyz.shape[0], xyz.shape[1]), float('inf'), device=xyz.device)
    indices = torch.zeros((xyz.shape[0], npoint), dtype=torch.long, device=xyz.device)
    farthest = torch.randint(0, xyz.shape[1], (xyz.shape[0],), device=xyz.device, dtype=torch.long)
    for i in range(npoint):
        indices[:, i] = farthest
        centroid = xyz[torch.arange(xyz.shape[0], device=xyz.device), farthest]  # (B, 3)
        new_dist = ((xyz - centroid.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, N)
        distances = torch.minimum(distances, new_dist)
        farthest = distances.argmax(dim=-1)
    return indices
