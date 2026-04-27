"""Ball Query — auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Ball Query: for each query point, find up to `nsample` source points
    within a ball of given `radius`.

    Unlike k-NN (which always returns exactly k neighbors), ball query returns
    neighbors based on spatial proximity — all points within the radius are
    candidates. This gives variable-density neighborhoods, which is important
    for PointNet++ to handle non-uniform point densities.

    Args:
        radius:  float, ball radius in Euclidean space.
                 Points with distance > radius from the query point are excluded.
        nsample: int, maximum number of neighbors to return per query point.
                 If a query point has fewer than nsample neighbors within the
                 radius, the last valid neighbor index is repeated to fill.
        xyz:     (B, N, 3) float tensor — source point cloud.
                 These are the points we search among.
        new_xyz: (B, M, 3) float tensor — query point cloud.
                 These are the centers of the balls.

    Returns:
        idx: (B, M, nsample) int32 tensor.
             idx[b, m, k] is the index into xyz[b, :, :] of the k-th neighbor
             of query point new_xyz[b, m, :].
             Values are in [0, N). If fewer than nsample neighbors were found,
             the remaining slots are filled with the first found neighbor index
             (NOT N or -1 — must be valid indices for downstream gather ops).

    Example:
        >>> xyz = torch.rand(2, 1024, 3)
        >>> new_xyz = xyz[:, :256, :]        # 256 query points (e.g. from FPS)
        >>> idx = ball_query(0.2, 32, xyz, new_xyz)  # → (2, 256, 32)
        >>> # idx[b, m, :] gives up to 32 neighbors of new_xyz[b, m] within r=0.2
    """
    if CUDA_AVAILABLE:
        from cuda_ops.ball_query import ball_query_cuda
        return ball_query_cuda(radius, nsample, xyz, new_xyz)
    return _ball_query_pytorch(radius, nsample, xyz, new_xyz)


def _ball_query_pytorch(radius, nsample, xyz, new_xyz):
    """Pure PyTorch ball query fallback.

    Algorithm:
      1. Compute pairwise distances between all query and source points:
           dists = torch.cdist(new_xyz, xyz)  → (B, M, N)
         This is a batched (B, M, N) distance matrix.
         dists[b, m, n] = ||new_xyz[b, m] - xyz[b, n]||_2

      2. For each query point m, we need up to nsample points where
         dists[b, m, n] <= radius. Approach:
         a. Sort by distance: dists.argsort(dim=-1) → sorted indices
         b. Take the first nsample: [:, :, :nsample]
         c. Check which are actually within radius:
              mask = sorted_dists > radius
         d. Replace out-of-radius indices with the first valid one:
              idx = torch.where(mask, first_valid_idx, sorted_idx)
         This gives a valid index for every slot — either a true neighbor
         or the nearest point as padding.

    Alternative approach (closer to the CUDA kernel):
      - For each query point, iterate all N source points
      - Collect indices where dist <= radius
      - Stop once we have nsample neighbors
      - Pad if needed

    Hints:
      - torch.cdist is O(B*M*N) in memory — fine for N<=2048 on CPU
      - Use .argsort() or .topk() for sorting
      - The padding strategy matters: always use a valid index (e.g. the
        nearest neighbor) so downstream gather ops don't crash
    """
    dists = torch.cdist(new_xyz, xyz)  # (B, M, N)
    sorted_dists, sorted_idx = dists.sort(dim=-1)  # (B, M, N)
    idx = sorted_idx[:, :, :nsample]  # (B, M, nsample)
    mask = sorted_dists[:, :, :nsample] > radius  # (B, M, nsample) bool
    first_valid_idx = idx[:, :, 0:1].expand(-1, -1, nsample)  # (B, M, nsample)
    idx = torch.where(mask, first_valid_idx, idx)  # (B, M, nsample)
    return idx

