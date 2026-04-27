"""Three-NN Interpolation — auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def three_nn(unknown, known):
    """
    Find the 3 nearest neighbors in `known` for each point in `unknown`.

    Used in PointNet++ Feature Propagation (FP) layers to upsample features
    from a coarser point set to a denser one. We only need 3 neighbors
    (not k-NN for arbitrary k) because the interpolation uses inverse-distance
    weighting with 3 neighbors — enough for smooth upsampling.

    Args:
        unknown: (B, N, 3) float tensor — query points (denser set).
                 N > M typically — these are the points we want to interpolate TO.
        known:   (B, M, 3) float tensor — reference points (coarser set).
                 These are the points we interpolate FROM.

    Returns:
        dist: (B, N, 3) float tensor — Euclidean distances to the 3 nearest
              known points for each unknown point.
              dist[b, n, k] = ||unknown[b, n] - known[b, idx[b, n, k]]||_2
        idx:  (B, N, 3) int32 tensor — indices of the 3 nearest known points.
              idx[b, n, k] ∈ [0, M). The 3 indices are ordered by distance
              (closest first).

    Example:
        >>> known   = torch.rand(2, 128, 3)    # 128 coarse points
        >>> unknown = torch.rand(2, 512, 3)    # 512 dense points
        >>> dist, idx = three_nn(unknown, known)
        >>> dist.shape  # (2, 512, 3) — 3 distances per query point
        >>> idx.shape   # (2, 512, 3) — 3 neighbor indices
    """
    if CUDA_AVAILABLE:
        from cuda_ops.interpolate import three_nn_cuda
        return three_nn_cuda(unknown, known)
    return _three_nn_pytorch(unknown, known)


def _three_nn_pytorch(unknown, known):
    """Pure PyTorch three_nn fallback.

    Algorithm:
      1. Compute all pairwise squared distances:
           dists = torch.cdist(unknown, known)  → (B, N, M)
         torch.cdist computes Euclidean distance, not squared.

      2. Find top-3 smallest distances:
           dists.topk(3, dim=-1, largest=False)
         Returns (values, indices) both of shape (B, N, 3).
         values[b, n, :] are the 3 smallest distances (sorted ascending).
         indices[b, n, :] are the corresponding indices into known.

      3. Return the distances and indices.
         Note: cdist already returns sqrt distances, so no need to sqrt again.
         But if you computed squared distances manually, you'd need sqrt.

    Hint:
      This is essentially a one-liner with torch.cdist + topk.
    """
    # TODO: Implement three_nn
    dists = torch.cdist(unknown, known)  # (B, N, M)
    dist, idx = dists.topk(3, dim=-1, largest=False)
    return dist, idx
    


def three_interpolate(features, idx, weight):
    """
    Weighted interpolation from 3 neighbors (inverse-distance weighting).

    For each unknown point n, we have 3 neighbor indices idx[b, n, :] and
    corresponding weights weight[b, n, :] (which sum to 1). The interpolated
    feature is:

        out[b, c, n] = Σ_k  weight[b, n, k] * features[b, c, idx[b, n, k]]

    Args:
        features: (B, C, M) float tensor — features at known (coarse) points.
                  C = feature channels (e.g. 256, 512, 1024).
        idx:      (B, N, 3) int tensor — 3 nearest neighbor indices from three_nn.
                  idx[b, n, k] ∈ [0, M).
        weight:   (B, N, 3) float tensor — interpolation weights.
                  weight[b, n, :].sum() ≈ 1.0 for each (b, n).
                  Typically inverse-distance: w_k = (1/d_k) / Σ(1/d_j).

    Returns:
        out: (B, C, N) float tensor — interpolated features at unknown points.
             out[b, c, n] = Σ_k weight[b, n, k] * features[b, c, idx[b, n, k]]

    Example:
        >>> features = torch.rand(2, 256, 128)   # 256-dim features at 128 pts
        >>> idx = torch.randint(0, 128, (2, 512, 3))  # 3 neighbors for 512 pts
        >>> weight = torch.rand(2, 512, 3)
        >>> weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize
        >>> out = three_interpolate(features, idx, weight)  # → (2, 256, 512)
    """
    if CUDA_AVAILABLE:
        from cuda_ops.interpolate import three_interpolate_cuda
        return three_interpolate_cuda(features, idx, weight)
    return _three_interpolate_pytorch(features, idx, weight)


def _three_interpolate_pytorch(features, idx, weight):
    """Pure PyTorch three_interpolate fallback.

    Algorithm:
      Input:  features (B, C, M), idx (B, N, 3), weight (B, N, 3)
      Output: out (B, C, N)

      Step 1 — Gather features at neighbor indices:
        We need: gathered[b, c, n, k] = features[b, c, idx[b, n, k]]
        - Expand idx from (B, N, 3) to (B, C, N, 3):
          idx.unsqueeze(1).expand(B, C, N, 3)  — no memory copy
        - Flatten for gather: reshape to (B, C, N*3)
        - features.gather(-1, flat_idx) → (B, C, N*3)
        - Reshape back to (B, C, N, 3)

      Step 2 — Weighted sum:
        - Expand weight from (B, N, 3) to (B, 1, N, 3) for broadcasting
        - (gathered * weight_expanded).sum(dim=-1) → (B, C, N)

    The whole thing is a gather + weighted reduction.

    Gotcha:
      features.gather(dim=-1, index) requires index values in [0, M).
      Make sure idx doesn't contain out-of-range values.
    """
    # TODO: Implement three_interpolate
    idx = idx.unsqueeze(1).expand(-1, features.shape[1], -1, -1)  # (B, C, N, 3)
    idx_flat = idx.reshape(features.shape[0], features.shape[1], -1)  # (B, C, N*3)
    gathered = features.gather(dim=-1, index=idx_flat)  # (B, C, N*3)
    gathered = gathered.reshape(features.shape[0], features.shape[1], idx.shape[2], idx.shape[3]) # (B, C, N, 3)
    weight_expanded = weight.unsqueeze(1)  # (B, 1, N, 3)
    out = (gathered * weight_expanded).sum(dim=-1)  # (B, C, N)
    return out


def three_interpolation(known_xyz, known_feat, unknown_xyz):
    """
    Full interpolation pipeline: three_nn → compute inverse-distance weights
    → three_interpolate.

    This is the high-level function used in PointNet++ FP layers. It takes
    channel-first coordinates and features (as used throughout the model)
    and handles the coordinate transposition internally.

    Args:
        known_xyz:   (B, 3, M) float tensor — coordinates of known (coarse) points.
                     Channel-first format as produced by SA layers.
        known_feat:  (B, C, M) float tensor — features at known points.
                     C = feature dimension from the coarser SA layer.
        unknown_xyz: (B, 3, N) float tensor — coordinates of unknown (dense) points.
                     N > M — these are the points we want to upsample features to.

    Returns:
        (B, C, N) float tensor — features interpolated from known to unknown points.

    Weight computation:
        dist, idx = three_nn(unknown, known)     # find 3 nearest neighbors
        w_k = 1 / (d_k + eps)                    # inverse distance
        weight = w / w.sum(dim=-1, keepdim=True)  # normalize to sum=1

    This function is already implemented — you only need to implement
    _three_nn_pytorch and _three_interpolate_pytorch.
    """
    # Transpose from channel-first (B, 3, N/M) to (B, N/M, 3)
    known_t = known_xyz.transpose(1, 2).contiguous()
    unknown_t = unknown_xyz.transpose(1, 2).contiguous()

    dist, idx = three_nn(unknown_t, known_t)

    # Inverse-distance weighting
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)

    return three_interpolate(known_feat, idx, weight)
