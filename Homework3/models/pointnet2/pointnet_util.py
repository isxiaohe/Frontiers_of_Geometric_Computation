"""PointNet++ building blocks: Set Abstraction (SA) and SA-MSG.

Hierarchical feature learning for point cloud classification.
  - SA layers: FPS downsample + ball query group + shared MLP + max-pool
  - SA-MSG: Multi-scale grouping for robustness to non-uniform density

Data flow (SSG classification):
  Input: (B, N, 3) point cloud
    → SA1: N→512 pts,  r=0.2, k=32, MLP [64,64,128]      → (B, 128, 512)
    → SA2: 512→128 pts, r=0.4, k=64, MLP [128,128,256]    → (B, 256, 128)
    → SA3: 128→1 pt,    global,     MLP [256,512,1024]     → (B, 1024, 1)
    → FC head: 1024 → 512 → 256 → num_classes
"""
import torch
import torch.nn as nn

from ops import furthest_point_sample, ball_query


# ─── Helper: sample_and_group ───────────────────────────────────────────
#
# This is the core spatial operation shared by all SA variants:
#   1. FPS to get npoint query points
#   2. Ball query to find neighbors
#   3. Compute relative coordinates (neighbor - centroid)
#   4. Concatenate features with relative xyz
#
# You need to implement this first — it's used by SA and SA_MSG.
# ────────────────────────────────────────────────────────────────────────

def sample_and_group(npoint, radius, nsample, xyz, points, group_all=False):
    """
    FPS downsampling + ball query grouping.

    Args:
        npoint:    int, number of query points to sample via FPS.
                   Set to None if group_all=True.
        radius:    float, ball radius for neighbor search.
        nsample:   int, max neighbors per query point.
        xyz:       (B, N, 3) point coordinates.
        points:    (B, C, N) per-point features from previous layer, or None
                   for the first layer (where only xyz coordinates are used).
        group_all: bool, if True, group ALL points into one set (used for the
                   final global SA layer). npoint is effectively 1.

    Returns:
        new_xyz: (B, npoint, 3) or (B, 1, 3) if group_all query point coords.
        new_points: (B, C+3, npoint, nsample) grouped features.
    """
    # TODO: Implement sample_and_group
    if group_all:
        # Group all points into one set
        new_xyz = torch.zeros(xyz.size(0), 1, 3, device=xyz.device)  # (B, 1, 3)
        grouped_xyz = xyz.unsqueeze(2)  # (B, N, 1, 3)
        if points is not None:
            new_points = torch.cat([grouped_xyz.permute(0, 3, 2, 1), points.unsqueeze(2)], dim=1)  # (B, C+3, 1, N)
        else:
            new_points = grouped_xyz.permute(0, 3, 2, 1)  # (B, 3, 1, N)
    else:
        # 1. FPS to sample npoint query points
        idx = furthest_point_sample(xyz, npoint)  # (B, npoint)
        new_xyz = xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)

        # 2. Ball query to find neighbors
        bidx = ball_query(radius, nsample, xyz, new_xyz).long()  # (B, npoint, nsample)

        # 3. Group xyz and compute relative coordinates
        B = xyz.shape[0]
        bidx_flat = bidx.reshape(B, -1)  # (B, npoint*nsample)
        grouped_xyz = xyz.gather(1, bidx_flat.unsqueeze(-1).expand(-1, -1, 3)).contiguous()  # (B, npoint*nsample, 3)
        grouped_xyz = grouped_xyz.reshape(B, npoint, nsample, 3)  # (B, npoint, nsample, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)

        if points is not None:
            # 4. Group features and concatenate with relative xyz
            C = points.shape[1]
            grouped_points = points.transpose(1, 2).gather(1, bidx_flat.unsqueeze(-1).expand(-1, -1, C)).contiguous()  # (B, npoint*nsample, C)
            grouped_points = grouped_points.reshape(B, npoint, nsample, C)  # (B, npoint, nsample, C)
            new_points = torch.cat([grouped_xyz.permute(0, 3, 1, 2).contiguous(), grouped_points.permute(0, 3, 1, 2).contiguous()], dim=1)  # (B, C+3, npoint, nsample)
        else:
            new_points = grouped_xyz.permute(0, 3, 1, 2).contiguous()  # (B, 3, npoint, nsample)

    return new_xyz, new_points


# ─── Set Abstraction (single-scale) ─────────────────────────────────────

class PointNetSetAbstraction(nn.Module):
    """SA layer with single-scale grouping.

    Architecture per local patch:
      Input: (B, in_channel + 3, npoint, nsample)
        → Conv2d(in_channel+3, mlp[0], 1) → BN2d → ReLU
        → Conv2d(mlp[0], mlp[1], 1)       → BN2d → ReLU
        → ...
        → Conv2d(mlp[-2], mlp[-1], 1)     → BN2d → ReLU
        → MaxPool over nsample dimension
      Output: (B, mlp[-1], npoint)

    Note: Conv2d with kernel_size=1 is equivalent to a shared MLP applied
    independently to each (npoint, nsample) location. The nsample dimension
    is like a spatial dimension that we max-pool over at the end.

    Args:
        npoint:    int, number of points to downsample to (via FPS).
                   None if group_all=True.
        radius:    float, ball radius.
        nsample:   int, max neighbors.
        in_channel: int, number of input feature channels (C from previous layer).
                    +3 for relative xyz coordinates.
        mlp:       list[int], MLP hidden dimensions, e.g. [64, 64, 128].
        group_all: bool, if True, groups all points (for final global layer).
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # Build MLP layers: Conv2d → BN2d → ReLU, applied to each local patch
        # Input channels = in_channel (features) + 3 (relative xyz)
        mlp_spec = [in_channel + 3] + list(mlp)
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        Args:
            xyz:    (B, N, 3) point coordinates.
            points: (B, C, N) per-point features, or None for first layer.

        Returns:
            new_xyz:    (B, npoint, 3) downsampled point coordinates.
            new_points: (B, mlp[-1], npoint) local features after MLP + max-pool.

        Steps:
          1. Call sample_and_group(...) to get grouped features
          2. Apply self.mlp (Conv2d layers) to the grouped features
          3. Max-pool over the nsample dimension (dim=-1)
          4. Return downsampled xyz and pooled features
        """
        # TODO: Implement SA forward
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.group_all)
        new_points = self.mlp(new_points)  # (B, mlp[-1], npoint, nsample)
        new_points = torch.max(new_points, dim=-1)[0]  # Max-pool over nsample → (B, mlp[-1], npoint)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """SA layer with Multi-Scale Grouping (MSG).

    Instead of one ball query, we use multiple radii simultaneously.
    For each scale, we independently group → MLP → max-pool, then concatenate
    all scale features. This makes the model robust to varying point densities.

    Example: radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128]
      - Scale 0: r=0.1, k=16  → MLP [32,32,64]   → (B, 64, npoint)
      - Scale 1: r=0.2, k=32  → MLP [64,64,128]   → (B, 128, npoint)
      - Scale 2: r=0.4, k=128 → MLP [64,96,128]   → (B, 128, npoint)
      - Concat all → (B, 64+128+128, npoint) = (B, 320, npoint)

    Args:
        npoint:      int, number of FPS-sampled points.
        radius_list: list[float], ball radii for each scale.
        nsample_list: list[int], max neighbors for each scale.
        in_channel:  int, input feature channels (+3 for relative xyz per scale).
        mlp_list:    list[list[int]], MLP dims for each scale.
    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        # Build one MLP per scale
        self.mlp_modules = nn.ModuleList()
        for i in range(len(radius_list)):
            mlp_spec = [in_channel + 3] + list(mlp_list[i])
            layers = []
            for j in range(len(mlp_spec) - 1):
                layers.append(nn.Conv2d(mlp_spec[j], mlp_spec[j + 1], 1, bias=False))
                layers.append(nn.BatchNorm2d(mlp_spec[j + 1]))
                layers.append(nn.ReLU())
            self.mlp_modules.append(nn.Sequential(*layers))

    def forward(self, xyz, points):
        """
        Args:
            xyz:    (B, N, 3) point coordinates.
            points: (B, C, N) per-point features, or None.

        Returns:
            new_xyz:    (B, npoint, 3) downsampled point coordinates.
            new_points: (B, sum_of_all_mlp_last_dims, npoint) concatenated multi-scale features.

        Steps:
          1. FPS to get self.npoint query points: new_xyz
          2. For each scale i:
             a. Ball query with radius_list[i], nsample_list[i]
             b. Group neighbors, compute relative xyz, concat features
             c. Apply self.mlp_modules[i] → max-pool over nsample dim
          3. Concatenate all scale outputs along the channel dimension
          4. Return new_xyz and concatenated features

        Hint: You can reuse the grouping logic from sample_and_group,
              but do it per-scale instead of calling the function directly.
              Or call sample_and_group once per scale with different radius/nsample.
        """
        # TODO: Implement SA-MSG forward
        new_xyz_indices = furthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = xyz.gather(1, new_xyz_indices.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)
        B = xyz.shape[0]
        new_points_list = []
        for radius, nsample, mlp_module in zip(self.radius_list, self.nsample_list, self.mlp_modules):
            grouping_idx = ball_query(radius, nsample, xyz, new_xyz).long()  # (B, npoint, nsample)
            gidx_flat = grouping_idx.reshape(B, -1)  # (B, npoint*nsample)
            grouped_xyz = xyz.gather(1, gidx_flat.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
            grouped_xyz = grouped_xyz.reshape(B, self.npoint, nsample, 3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)
            if points is not None:
                C = points.shape[1]
                grouped_points = points.transpose(1, 2).gather(1, gidx_flat.unsqueeze(-1).expand(-1, -1, C)).contiguous()
                grouped_points = grouped_points.reshape(B, self.npoint, nsample, C)
                grouped_features = torch.cat([grouped_xyz.permute(0, 3, 1, 2).contiguous(), grouped_points.permute(0, 3, 1, 2).contiguous()], dim=1)
            else:
                grouped_features = grouped_xyz.permute(0, 3, 1, 2).contiguous()
            new_points = mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
            new_points = torch.max(new_points, dim=-1)[0]  # Max-pool over nsample → (B, mlp[-1], npoint)
            new_points_list.append(new_points)
        new_points_concat = torch.cat(new_points_list, dim=1)  # Concat all scales → (B, sum_of_all_mlp_last_dims, npoint)
        return new_xyz, new_points_concat
            
