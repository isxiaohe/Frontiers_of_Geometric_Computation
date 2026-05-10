"""PointNeXt classification models (S and B variants).

Architecture overview:

  PointNeXt = Stem SA → [InvResMLP blocks + SA downsample] × stages → Global SA → FC head

  Key innovation: InvResMLP (Inverted Residual MLP), inspired by MobileNetV2:
    - Expand (1x1 Conv, narrow → wide)
    - Local Aggregation (ball query group + MLP + max-pool in expanded space)
    - Project (1x1 Conv, wide → narrow)
    - Skip connection (residual add)

  PointNeXt-S (~1.4M params, expand_ratio=2):
    Stem:  N→512, r=0.15, k=32 → 64ch
    Stage 1: InvResMLP×1 (64→64, r=0.15) + SA 512→128 (r=0.2, 64→128)
    Stage 2: InvResMLP×1 (128→128, r=0.2) + SA 128→32  (r=0.4, 128→256)
    Stage 3: InvResMLP×1 (256→256, r=0.4) + Global SA     (256→512)
    Head: 512 → 256 → 40

  PointNeXt-B (~3.5M params, expand_ratio=4, 2 blocks per stage):
    Same structure but 2 InvResMLP blocks per stage and expand_ratio=4.

Data flow:
  Input: (B, N, 3)
    → Stem SA: FPS N→512, group+MLP+maxpool → (B, 64, 512), xyz=(B, 512, 3)
    → Stage 1: InvResMLP (same 512 pts) + SA downsample → (B, 128, 128), xyz=(B, 128, 3)
    → Stage 2: InvResMLP (same 128 pts) + SA downsample → (B, 256, 32),  xyz=(B, 32, 3)
    → Stage 3: InvResMLP (same 32 pts)  + Global SA      → (B, 512)
    → FC head → (B, num_classes)

Reference: Qian et al., "PointNeXt: Revisiting PointNet++ with Improved Training
and Scaling Strategies", NeurIPS 2022.
"""
import torch
import torch.nn as nn

from ops import furthest_point_sample, ball_query


# ─── LocalAggregation ──────────────────────────────────────────────────
#
# Shared building block for grouping neighbors + shared MLP + max-pool.
# Used by both InvResMLP (same-resolution feature mixing) and SA downsample.
#
# This is the same pattern as PointNet++ sample_and_group + MLP + max-pool,
# but packaged as a standalone module that takes pre-computed query points.
# ────────────────────────────────────────────────────────────────────────

class LocalAggregation(nn.Module):
    """Group neighbors + shared MLP + max-pool.

    For each query point in new_xyz:
      1. Ball query to find nsample neighbors within radius
      2. Gather neighbor features, compute relative xyz (neighbor - centroid)
      3. Concatenate [relative_xyz, gathered_features]
      4. Apply shared MLP (Conv2d with kernel_size=1)
      5. Max-pool over neighbor dimension

    Args:
        in_channel: int, input feature channels C.
        mlp:        list[int], MLP hidden dims. Output = mlp[-1] channels.
        radius:     float, ball query radius.
        nsample:    int, max neighbors per query point.
    """

    def __init__(self, in_channel, mlp, radius, nsample):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

        # Build MLP: Conv2d(in_channel+3, mlp[0], 1) → BN2d → ReLU → ...
        # Input channels = in_channel (features) + 3 (relative xyz)
        mlp_spec = [in_channel + 3] + list(mlp)
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, new_xyz, features):
        """
        Args:
            xyz:      (B, N, 3) all point coordinates.
            new_xyz:  (B, npoint, 3) query point coordinates.
            features: (B, C, N) per-point features.

        Returns:
            (B, mlp[-1], npoint) aggregated features.

        Steps:
          1. Ball query:
             idx = ball_query(radius, nsample, xyz, new_xyz).long()
             → (B, npoint, nsample) neighbor indices

          2. Gather xyz of neighbors and compute relative coordinates:
             Flatten idx to (B, npoint*nsample), gather from xyz,
             reshape to (B, npoint, nsample, 3),
             subtract new_xyz → relative xyz

          3. Gather features of neighbors:
             Use same flattened idx, gather from features (B, C, N),
             reshape to (B, npoint, nsample, C)

          4. Concatenate [relative_xyz, gathered_features]:
             Both need to be permuted to (B, C+3, npoint, nsample) format
             → concat along channel dim → (B, C+3, npoint, nsample)

          5. Apply self.mlp → (B, mlp[-1], npoint, nsample)

          6. Max-pool over last dim (nsample) → (B, mlp[-1], npoint)

        Hints:
          - This is the same pattern as sample_and_group() + MLP + max-pool
            in models/pointnet2/pointnet_util.py — refer to that for reference.
          - Use .contiguous() after gather and permute (MPS backward compatibility).
          - Use .reshape() not .view() (MPS compatibility).
          - Ball query returns (B, npoint, nsample) indices.
            Flatten to (B, npoint*nsample) before using with .gather().
        """
        B = xyz.shape[0]
        npoint = new_xyz.shape[1]

        # 1. Ball query → neighbor indices
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz).long()  # (B, npoint, nsample)

        # 2. Gather neighbor xyz and compute relative coordinates
        idx_flat = idx.reshape(B, -1)  # (B, npoint*nsample)
        grouped_xyz = xyz.gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous().reshape(B, npoint, self.nsample, 3)  # (B, npoint, nsample, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # relative xyz

        # 3. Gather neighbor features
        C = features.shape[1]
        grouped_feats = features.transpose(1, 2).gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).contiguous().reshape(B, npoint, self.nsample, C)  # (B, npoint, nsample, C)

        # 4. Concat [relative_xyz, features] → (B, C+3, npoint, nsample)
        grouped = torch.cat([
            grouped_xyz.permute(0, 3, 1, 2).contiguous(),
            grouped_feats.permute(0, 3, 1, 2).contiguous(),
        ], dim=1)

        # 5. MLP → (B, mlp[-1], npoint, nsample)
        # 6. Max-pool over nsample → (B, mlp[-1], npoint)
        return torch.max(self.mlp(grouped), dim=-1)[0]


# ─── InvResMLP (Inverted Residual MLP) ─────────────────────────────────
#
# PointNeXt's core architectural contribution.
# Inspired by MobileNetV2: expand → local processing → project + skip.
#
# The key insight: doing expensive local aggregation in a higher-dimensional
# expanded space gives more representational power per parameter.
# ────────────────────────────────────────────────────────────────────────

class InvResMLP(nn.Module):
    """Inverted Residual MLP block.

    Architecture:
      features (C_in)
        → expand:  Conv1d(C_in → expanded) + BN + ReLU
        → local aggregation in expanded space (group + MLP + max-pool)
        → project: Conv1d(expanded → C_out) + BN (no ReLU here!)
        → output = shortcut(features) + projected

    Operates at SAME resolution — no FPS downsampling inside this block.
    Downsampling happens in separate SA layers between stages.

    Args:
        in_channel:   int, input feature channels.
        out_channel:  int, output feature channels.
        expand_ratio: int, expansion factor (expanded = in_channel * expand_ratio).
        radius:       float, ball query radius for local aggregation.
        nsample:      int, max neighbors for local aggregation.
    """

    def __init__(self, in_channel, out_channel, expand_ratio=2,
                 radius=0.2, nsample=32):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        expanded = in_channel * expand_ratio

        # Expand: narrow → wide (Conv1d, shared across all N points)
        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channel, expanded, 1, bias=False),
            nn.BatchNorm1d(expanded),
            nn.ReLU(),
        )

        # Local aggregation in expanded space
        # Input: expanded + 3 (relative xyz) → single-layer MLP → expanded
        self.local_agg = LocalAggregation(
            in_channel=expanded,
            mlp=[expanded],
            radius=radius,
            nsample=nsample,
        )

        # Project: wide → narrow (NO ReLU — linear bottleneck, like MobileNetV2)
        self.project_conv = nn.Sequential(
            nn.Conv1d(expanded, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

        # Skip connection: identity if in==out, else 1x1 projection
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm1d(out_channel),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3) point coordinates (unchanged).
            features: (B, C_in, N) per-point features.

        Returns:
            (B, C_out, N) updated features. Same N, no downsampling.

        Steps:
          1. Save shortcut: residual = self.shortcut(features)
          2. Expand:  x = self.expand_conv(features)        → (B, expanded, N)
          3. Aggregate: x = self.local_agg(xyz, xyz, x)
             Note: both xyz args are the SAME — query points = all points
             (no downsampling, we aggregate neighbors for every point)
             → (B, expanded, N)
          4. Project: x = self.project_conv(x)              → (B, C_out, N)
          5. Add skip: output = residual + x
          6. Return output
        """
        # 1. Save shortcut
        residual = self.shortcut(features)

        # 2. Expand: narrow → wide
        x = self.expand_conv(features)  # (B, expanded, N)

        # 3. Local aggregation (same xyz for query and all points — no downsampling)
        x = self.local_agg(xyz, xyz, x)  # (B, expanded, N)

        # 4. Project: wide → narrow (no ReLU — linear bottleneck)
        x = self.project_conv(x)  # (B, C_out, N)

        # 5. Skip connection
        return residual + x


# ─── PointNeXtSA (downsampling SA layer) ────────────────────────────────
#
# FPS downsample + LocalAggregation. Used between stages to reduce
# the number of points while extracting features.
# ────────────────────────────────────────────────────────────────────────

class PointNeXtSA(nn.Module):
    """Downsampling Set Abstraction layer.

    FPS to select npoint query points, then LocalAggregation to extract features.

    Args:
        npoint:     int, number of points after downsampling.
        radius:     float, ball query radius.
        nsample:    int, max neighbors.
        in_channel: int, input feature channels.
        out_channel: int, output feature channels.
    """

    def __init__(self, npoint, radius, nsample, in_channel, out_channel):
        super().__init__()
        self.npoint = npoint
        self.local_agg = LocalAggregation(
            in_channel=in_channel,
            mlp=[out_channel, out_channel],
            radius=radius,
            nsample=nsample,
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, C, N)

        Returns:
            new_xyz:      (B, npoint, 3)
            new_features: (B, out_channel, npoint)
        """
        # 1. FPS downsample
        idx = furthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = xyz.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous()  # (B, npoint, 3)

        # 2. Local aggregation at new points
        new_features = self.local_agg(xyz, new_xyz, features)

        return new_xyz, new_features


# ─── GlobalSA ──────────────────────────────────────────────────────────
#
# Final global pooling: group ALL points → MLP → max-pool → global vector.
# No FPS needed — every point contributes to the single global feature.
# ────────────────────────────────────────────────────────────────────────

class GlobalSA(nn.Module):
    """Global set abstraction: group ALL points → MLP → max-pool → vector."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        mlp_spec = [in_channel + 3, out_channel, out_channel]
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, C, N)
        Returns:
            (B, out_channel) global feature vector.
        """
        B = xyz.shape[0]
        # All points grouped together — relative xyz = original xyz (center at origin)
        grouped_xyz = xyz.unsqueeze(2)  # (B, N, 1, 3)
        if features is not None:
            grouped = torch.cat([
                grouped_xyz.permute(0, 3, 2, 1).contiguous(),  # (B, 3, 1, N)
                features.unsqueeze(2),                          # (B, C, 1, N)
            ], dim=1)  # (B, C+3, 1, N)
        else:
            grouped = grouped_xyz.permute(0, 3, 2, 1).contiguous()

        x = self.mlp(grouped)         # (B, out_channel, 1, N)
        x = torch.max(x, dim=-1)[0]   # (B, out_channel, 1)
        x = x.reshape(B, -1)          # (B, out_channel)
        return x


# ─── PointNeXt Classification Model ────────────────────────────────────

class PointNeXtCls(nn.Module):
    """PointNeXt classification model.

    Pipeline:
      Input (B, N, 3)
        → Stem SA (FPS N→512, embed to channels[0])
        → Stage 1: InvResMLP blocks + SA downsample (512→128)
        → Stage 2: InvResMLP blocks + SA downsample (128→32)
        → Stage 3: InvResMLP blocks + Global SA
        → FC classification head

    Args:
        num_classes: int, output classes (40 for ModelNet40).
        in_channel:  int, input feature channels (3 for xyz, 6 with normals).
        encoder_params: dict configuring the encoder:
            'channels':     list[int], e.g. [64, 128, 256, 512]
            'blocks':       list[int], InvResMLP blocks per stage, e.g. [1, 1, 1]
            'npoints':      list[int], points per stage, e.g. [512, 128, 32]
            'radii':        list[float], radius per stage, e.g. [0.15, 0.2, 0.4]
            'nsample':      int, neighbors per group (shared across stages)
            'expand_ratio': int, InvResMLP expansion factor
    """

    def __init__(self, num_classes=40, in_channel=3, encoder_params=None):
        super().__init__()
        ep = encoder_params or {}
        channels = ep.get('channels', [64, 128, 256, 512])
        blocks = ep.get('blocks', [1, 1, 1])
        npoints = ep.get('npoints', [512, 128, 32])
        radii = ep.get('radii', [0.15, 0.2, 0.4])
        nsample = ep.get('nsample', 32)
        expand_ratio = ep.get('expand_ratio', 2)

        # Stem: initial embedding SA, N → npoints[0]
        self.stem = PointNeXtSA(
            npoint=npoints[0],
            radius=radii[0],
            nsample=nsample,
            in_channel=in_channel,
            out_channel=channels[0],
        )

        # Per-stage: InvResMLP blocks (same resolution) + downsample (SA or Global)
        self.stage_blocks = nn.ModuleList()
        self.stage_downsample = nn.ModuleList()

        for i in range(len(blocks)):
            # InvResMLP blocks at current resolution
            self.stage_blocks.append(nn.ModuleList([
                InvResMLP(
                    in_channel=channels[i],
                    out_channel=channels[i],
                    expand_ratio=expand_ratio,
                    radius=radii[i],
                    nsample=nsample,
                )
                for _ in range(blocks[i])
            ]))

            # Downsample: SA for all stages except the last (which uses GlobalSA)
            if i < len(blocks) - 1:
                self.stage_downsample.append(PointNeXtSA(
                    npoint=npoints[i + 1],
                    radius=radii[i + 1],
                    nsample=nsample,
                    in_channel=channels[i],
                    out_channel=channels[i + 1],
                ))
            else:
                self.stage_downsample.append(GlobalSA(
                    in_channel=channels[i],
                    out_channel=channels[i + 1],
                ))

        # Classification head
        head_dim = channels[-1]  # 512
        self.head = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6) with normals.
        Returns:
            (B, num_classes) logits.
        """
        xyz_coords = xyz[:, :, :3].contiguous()
        if xyz.shape[-1] > 3:
            features = xyz[:, :, 3:].transpose(1, 2).contiguous()
        else:
            features = xyz_coords.transpose(1, 2).contiguous()

        # Stem: N → npoints[0]
        xyz_coords, features = self.stem(xyz_coords, features)

        # Stages: InvResMLP blocks + downsample
        for i in range(len(self.stage_blocks)):
            # InvResMLP blocks (same resolution)
            for block in self.stage_blocks[i]:
                features = block(xyz_coords, features)

            # Downsample (SA or Global)
            ds = self.stage_downsample[i]
            if isinstance(ds, GlobalSA):
                features = ds(xyz_coords, features)  # → (B, channels[-1])
            else:
                xyz_coords, features = ds(xyz_coords, features)

        return self.head(features)


# ─── Factory functions ──────────────────────────────────────────────────

def pointnext_s(num_classes=40, **kwargs):
    """PointNeXt-S: 3 stages, 1 block each, expand_ratio=2. ~1.4M params."""
    return PointNeXtCls(
        num_classes=num_classes,
        in_channel=3,
        encoder_params=dict(
            channels=[64, 128, 256, 512],
            blocks=[1, 1, 1],
            npoints=[512, 128, 32],
            radii=[0.15, 0.2, 0.4],
            nsample=32,
            expand_ratio=2,
        ),
        **kwargs,
    )


def pointnext_b(num_classes=40, **kwargs):
    """PointNeXt-B: 3 stages, 2 blocks each, expand_ratio=2. ~2.7M params."""
    return PointNeXtCls(
        num_classes=num_classes,
        in_channel=3,
        encoder_params=dict(
            channels=[64, 128, 256, 512],
            blocks=[2, 2, 2],
            npoints=[512, 128, 32],
            radii=[0.15, 0.2, 0.4],
            nsample=32,
            expand_ratio=2,
        ),
        **kwargs,
    )
