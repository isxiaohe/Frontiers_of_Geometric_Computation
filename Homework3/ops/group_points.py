"""Group Points (gather by index) — auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def group_points(features, idx):
    """
    Gather point features according to a neighbor index matrix.

    After ball_query gives us (B, M, nsample) neighbor indices, we need to
    actually collect the features of those neighbors. This is the "grouping"
    step in PointNet++ — it creates local patches of features around each
    query point, which are then processed by shared MLPs.

    Args:
        features: (B, C, N) float tensor — per-point feature vectors.
                  B = batch, C = feature channels, N = number of points.
                  C starts at 3 (xyz) and grows through the network
                  (e.g. 3 → 128 → 256 → 1024).
        idx:      (B, M, K) int tensor — neighbor indices from ball_query.
                  M = number of query points (e.g. from FPS).
                  K = number of neighbors per query (e.g. nsample=32).
                  idx[b, m, k] ∈ [0, N) — index into the N source points.

    Returns:
        grouped: (B, C, M, K) float tensor — gathered features.
                 grouped[b, c, m, k] = features[b, c, idx[b, m, k]]
                 This can be thought of as M local patches, each of size K,
                 with C feature channels.

    Example:
        >>> features = torch.rand(2, 64, 1024)   # 64-dim features at 1024 pts
        >>> idx = torch.randint(0, 1024, (2, 256, 32))  # 256 queries × 32 nbrs
        >>> grouped = group_points(features, idx)  # → (2, 64, 256, 32)
        >>> # grouped[b, :, m, :] is the local patch around query point m
        >>> # Can now apply Conv2d over (M, K) "spatial" dims
    """
    if CUDA_AVAILABLE:
        from cuda_ops.group_points import group_points_cuda
        return group_points_cuda(features, idx)
    return _group_points_pytorch(features, idx)


def _group_points_pytorch(features, idx):
    """Pure PyTorch group_points fallback.

    This is a batched gather operation along the point dimension.

    Algorithm:
        Input:  features (B, C, N), idx (B, M, K)
        Output: grouped (B, C, M, K)

      Step 1 — Prepare index for gather:
        idx starts as    (B, M, K)
        We need it as    (B, C, M*K)  to gather from features of shape (B, C, N)
        - idx.unsqueeze(1)            → (B, 1, M, K)
        - .expand(B, C, M, K)        → (B, C, M, K)  (no memory copy)
        - .reshape(B, C, M*K)        → (B, C, M*K)   (contiguous view)

      Step 2 — Gather:
        features.gather(dim=-1, index=flat_idx)  → (B, C, M*K)
        This picks features[b, c, flat_idx[b, c, :]] for each (b, c).

      Step 3 — Reshape:
        .reshape(B, C, M, K)  → (B, C, M, K)

    Key insight:
      torch.gather(input, dim, index) selects elements along `dim`.
      For dim=-1 on (B, C, N) input, index must be (B, C, ?) with values in [0, N).
      The index tensor must have the same leading dims as the input.

    Gotcha:
      .expand() is a view — it shares memory, so it's free.
      But .reshape() after .expand() may need .contiguous() — use
      .reshape() which handles this automatically (copies if needed).
    """
    # TODO: Implement group_points
    B, C, N = features.shape
    _, M, K = idx.shape

    flat_idx = idx.unsqueeze(1).expand(B, C, M, K).reshape(B, C, M * K).contiguous()
    grouped = features.gather(dim=-1, index=flat_idx)
    return grouped.reshape(B, C, M, K)