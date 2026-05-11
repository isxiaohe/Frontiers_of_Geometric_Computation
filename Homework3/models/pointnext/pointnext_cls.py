"""PointNeXt classification models (S and B variants).

Reference implementation matching the openpoints codebase:
  Qian et al., "PointNeXt: Revisiting PointNet++ with Improved Training
  and Scaling Strategies", NeurIPS 2022.

Architecture (PointNeXt-S):
  blocks=[1,1,1,1,1,1], strides=[1,2,2,2,2,1], width=32

  Input (B, N, 3)
    Stage 0 (stem/head, stride=1): Conv1d stem, 3→32, no grouping
    Stage 1 (stride=2): FPS N→N/2, SA(2-layer MLP+res) 32→64
    Stage 2 (stride=2): FPS →N/4, SA 64→128
    Stage 3 (stride=2): FPS →N/8, SA 128→256
    Stage 4 (stride=2): FPS →N/16, SA 256→512
    Stage 5 (stride=1): Global SA (group all), 512→512
    Head: 512→512→256→40

Key components:
  - LocalAggregation: ball query + normalized relative xyz + MLP + max-pool
  - SetAbstraction: FPS + LocalAggregation, multi-layer MLP, optional residual
  - InvResMLP: local agg → pwconv(expand→project) → skip → ReLU
  - Stem: simple Conv1d embedding (no grouping)

Training recipe (default.yaml):
  AdamW, lr=0.001, wd=0.05, cosine 600 epochs
  SmoothCrossEntropy (label_smoothing=0.2)
  Augmentation: ScaleAndTranslate (no rotation for ModelNet40)
"""
import torch
import torch.nn as nn

from ops import furthest_point_sample, ball_query


# ─── LocalAggregation ──────────────────────────────────────────────────

class LocalAggregation(nn.Module):
    """Group neighbors + MLP + max-pool.

    For each query point:
      1. Ball query to find neighbors within radius
      2. Compute relative xyz (normalized by radius)
      3. Concatenate [normalized_relative_xyz, gathered_features]
      4. Apply shared MLP (Conv2d)
      5. Max-pool over neighbor dimension

    Args:
        in_channel:  int, input feature channels C.
        mlp:         list[int], output channels per MLP layer.
        radius:      float, ball query radius.
        nsample:     int, max neighbors.
        normalize_dp: bool, divide relative xyz by radius.
    """

    def __init__(self, in_channel, mlp, radius, nsample, normalize_dp=True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.normalize_dp = normalize_dp

        # Build MLP: first layer gets in_channel + 3 (relative xyz)
        mlp_spec = [in_channel + 3] + list(mlp)
        layers = []
        for i in range(len(mlp_spec) - 1):
            layers.append(nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(mlp_spec[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, support_xyz, query_xyz, features):
        """
        Args:
            support_xyz: (B, N, 3) all point coordinates.
            query_xyz:   (B, npoint, 3) query point coordinates.
            features:    (B, C, N) per-point features.
        Returns:
            (B, mlp[-1], npoint) aggregated features.
        """
        B = query_xyz.shape[0]
        npoint = query_xyz.shape[1]

        # 1. Ball query
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz).long()
        idx_flat = idx.reshape(B, -1)  # (B, npoint*nsample)

        # 2. Gather neighbor xyz, compute normalized relative coords
        grouped_xyz = support_xyz.gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous().reshape(B, npoint, self.nsample, 3)
        grouped_xyz = grouped_xyz - query_xyz.unsqueeze(2)  # relative
        if self.normalize_dp:
            grouped_xyz = grouped_xyz / max(self.radius, 1e-8)

        # 3. Gather neighbor features
        C = features.shape[1]
        grouped_feats = features.transpose(1, 2).gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).contiguous().reshape(B, npoint, self.nsample, C)

        # 4. Concat [relative_xyz, features] → (B, C+3, npoint, nsample)
        grouped = torch.cat([
            grouped_xyz.permute(0, 3, 1, 2).contiguous(),
            grouped_feats.permute(0, 3, 1, 2).contiguous(),
        ], dim=1)

        # 5. MLP → max-pool
        return torch.max(self.mlp(grouped), dim=-1)[0]


# ─── SetAbstraction ────────────────────────────────────────────────────

class SetAbstraction(nn.Module):
    """Set Abstraction with multi-layer MLP and optional residual.

    Three modes:
      - is_head=True:  Conv1d stem (no grouping, no pooling)
      - stride=1 (not head): global aggregation (group all points)
      - stride>1: FPS downsample + local aggregation

    Args:
        in_channels:  int, input channels.
        out_channels: int, output channels.
        layers:       int, number of MLP layers.
        stride:       int, downsampling ratio (1=no downsample).
        radius:       float, ball query radius.
        nsample:      int, max neighbors.
        use_res:      bool, use residual connection.
        is_head:      bool, stem Conv1d mode (no grouping).
        normalize_dp: bool, normalize relative xyz by radius.
    """

    def __init__(self, in_channels, out_channels, layers=2, stride=1,
                 radius=0.15, nsample=32, use_res=False,
                 is_head=False, normalize_dp=True):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.is_global = not is_head and stride == 1
        self.use_res = use_res and stride > 1  # residual only for downsample
        self.radius = radius
        self.nsample = nsample
        self.normalize_dp = normalize_dp

        if is_head:
            # Stem: Conv1d only, no grouping
            self.convs = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            # Multi-layer MLP (Conv2d)
            # For stride>1: mid = out//2 (bottleneck); for stride=1: mid = out
            mid = out_channels // 2 if stride > 1 else out_channels
            mlp_ch = [in_channels] + [mid] * (layers - 1) + [out_channels]
            convs = []
            for i in range(len(mlp_ch) - 1):
                in_ch = mlp_ch[i] + 3 if i == 0 else mlp_ch[i]  # +3 for relative xyz
                out_ch = mlp_ch[i + 1]
                # No activation on last layer if using residual
                is_last = (i == len(mlp_ch) - 2)
                has_act = not (is_last and self.use_res)
                convs.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
                convs.append(nn.BatchNorm2d(out_ch))
                if has_act:
                    convs.append(nn.ReLU())
            self.convs = nn.Sequential(*convs)

        # Residual projection
        if self.use_res:
            if in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                self.skip = nn.Identity()
            self.act = nn.ReLU()

    def _group_and_agg(self, support_xyz, query_xyz, features):
        """Ball query + gather + concat + MLP + max-pool."""
        B, npoint, _ = query_xyz.shape
        N = support_xyz.shape[1]
        nsample = self.nsample
        radius = self.radius

        if self.is_global:
            # Group all: each query point sees ALL support points
            # Use ball query with very large radius and nsample=N
            nsample = N
            radius = 1e6

        idx = ball_query(radius, nsample, support_xyz, query_xyz).long()
        idx_flat = idx.reshape(B, -1)

        # Gather xyz → relative coords
        grouped_xyz = support_xyz.gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous().reshape(B, npoint, nsample, 3)
        grouped_xyz = grouped_xyz - query_xyz.unsqueeze(2)
        if self.normalize_dp:
            grouped_xyz = grouped_xyz / max(self.radius, 1e-8)

        # Gather features
        C = features.shape[1]
        grouped_feats = features.transpose(1, 2).gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, C)
        ).contiguous().reshape(B, npoint, nsample, C)

        # Concat + MLP + max-pool
        grouped = torch.cat([
            grouped_xyz.permute(0, 3, 1, 2).contiguous(),
            grouped_feats.permute(0, 3, 1, 2).contiguous(),
        ], dim=1)
        return torch.max(self.convs(grouped), dim=-1)[0]

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, C, N)
        Returns:
            new_xyz:      (B, npoint, 3) or (B, N, 3)
            new_features: (B, out_channels, npoint) or (B, out_channels)
        """
        B, N, _ = xyz.shape

        if self.is_head:
            # Stem Conv1d
            return xyz, self.convs(features)

        if self.is_global:
            # Global aggregation: no FPS, group all
            new_features = self._group_and_agg(xyz, xyz, features)
            return xyz, new_features

        # FPS downsample
        npoint = N // self.stride
        idx = furthest_point_sample(xyz, npoint).long()
        new_xyz = xyz.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous()

        # Save identity for residual
        if self.use_res:
            identity = self.skip(
                features.gather(-1, idx.unsqueeze(1).expand(-1, features.shape[1], -1))
            )

        # Group + aggregate
        new_features = self._group_and_agg(xyz, new_xyz, features)

        if self.use_res:
            new_features = self.act(new_features + identity)

        return new_xyz, new_features


# ─── InvResMLP ─────────────────────────────────────────────────────────

class InvResMLP(nn.Module):
    """Inverted Residual MLP block (PointNeXt's core for B+ variants).

    Architecture (matching openpoints):
      features (C)
        → local aggregation (group + MLP + pool): C → C
        → pointwise conv: C → expanded → C  (expand/project)
        → skip connection: output = identity + projected
        → ReLU activation

    Operates at SAME resolution — no FPS downsampling.

    Args:
        channels:    int, input/output channels (same).
        expansion:   int, expansion ratio for pwconv.
        radius:      float, ball query radius.
        nsample:     int, max neighbors.
        normalize_dp: bool, normalize relative xyz.
    """

    def __init__(self, channels, expansion=4, radius=0.15, nsample=32,
                 normalize_dp=True):
        super().__init__()
        mid = int(channels * expansion)

        # Local aggregation: channels → channels
        self.local_agg = LocalAggregation(
            in_channel=channels,
            mlp=[channels],
            radius=radius,
            nsample=nsample,
            normalize_dp=normalize_dp,
        )

        # Pointwise: channels → expanded → channels (no ReLU on last)
        self.pwconv = nn.Sequential(
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

        self.act = nn.ReLU()

    def forward(self, xyz, features):
        """
        Args:
            xyz:      (B, N, 3)
            features: (B, C, N)
        Returns:
            (B, C, N) updated features.
        """
        identity = features
        f = self.local_agg(xyz, xyz, features)  # C → C
        f = self.pwconv(f)                        # C → expanded → C
        f = f + identity                          # skip
        f = self.act(f)                           # ReLU after skip
        return xyz, f


# ─── PointNeXt Classification Model ────────────────────────────────────

class PointNeXtCls(nn.Module):
    """PointNeXt classification model.

    Matches the openpoints PointNextEncoder + ClsHead architecture.
    Each stage = [SetAbstraction (first)] + [InvResMLP × (blocks-1)]
    The first stage with stride=1 acts as stem (Conv1d, no grouping).
    The last stage with stride=1 does global aggregation.

    Args:
        num_classes:    int, output classes.
        in_channels:    int, input channels (3 for xyz).
        width:          int, initial channel width.
        blocks:         list[int], blocks per stage (includes SA).
        strides:        list[int], downsampling ratio per stage.
        radius:         float, initial ball query radius.
        radius_scaling: float, radius multiplier per downsample stage.
        nsample:        int, max neighbors.
        expansion:      int, InvResMLP expansion ratio.
        sa_layers:      int, MLP layers in SA blocks.
        sa_use_res:     bool, residual in SA blocks.
        normalize_dp:   bool, normalize relative xyz.
    """

    def __init__(self, num_classes=40, in_channels=3, width=32,
                 blocks=None, strides=None,
                 radius=0.15, radius_scaling=1.5, nsample=32,
                 expansion=4, sa_layers=2, sa_use_res=True,
                 normalize_dp=True):
        super().__init__()
        blocks = blocks or [1, 1, 1, 1, 1, 1]
        strides = strides or [1, 2, 2, 2, 2, 1]

        # Compute channels: double width on each stride>1
        channels = []
        w = width
        for s in strides:
            if s > 1:
                w *= 2
            channels.append(w)
        # channels: e.g. [32, 64, 128, 256, 512, 512]

        # Build encoder: list of (SA + optional InvResMLPs) per stage
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(len(blocks)):
            stage = nn.ModuleList()

            # SetAbstraction (first block of each stage)
            is_head = (i == 0 and strides[i] == 1)
            stage.append(SetAbstraction(
                in_channels=in_ch,
                out_channels=channels[i],
                layers=sa_layers if not is_head else 1,
                stride=strides[i],
                radius=radius,
                nsample=nsample,
                use_res=sa_use_res,
                is_head=is_head,
                normalize_dp=normalize_dp,
            ))
            in_ch = channels[i]

            # InvResMLP blocks (rest of blocks)
            for _ in range(1, blocks[i]):
                stage.append(InvResMLP(
                    channels=in_ch,
                    expansion=expansion,
                    radius=radius,
                    nsample=nsample,
                    normalize_dp=normalize_dp,
                ))

            self.encoder.append(stage)

            # Scale radius after each downsample stage
            if strides[i] > 1:
                radius *= radius_scaling

        out_channels = channels[-1]

        # Classification head: Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Linear
        self.head = nn.Sequential(
            nn.Linear(out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
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

        # Encoder stages
        for stage in self.encoder:
            for layer in stage:
                xyz_coords, features = layer(xyz_coords, features)

        # Global max-pool over remaining points
        features = torch.max(features, dim=-1)[0]  # (B, C)

        return self.head(features)


# ─── Factory functions ──────────────────────────────────────────────────

def pointnext_s(num_classes=40, **kwargs):
    """PointNeXt-S: blocks=[1,1,1,1,1,1], width=32, expansion=4. ~1.4M params."""
    return PointNeXtCls(
        num_classes=num_classes,
        in_channels=3,
        width=32,
        blocks=[1, 1, 1, 1, 1, 1],
        strides=[1, 2, 2, 2, 2, 1],
        radius=0.15,
        radius_scaling=1.5,
        nsample=32,
        expansion=4,
        sa_layers=2,
        sa_use_res=True,
        normalize_dp=True,
        **kwargs,
    )


def pointnext_b(num_classes=40, **kwargs):
    """PointNeXt-B: blocks=[1,2,3,2,2], width=32, expansion=4. ~4.5M params."""
    return PointNeXtCls(
        num_classes=num_classes,
        in_channels=3,
        width=32,
        blocks=[1, 2, 3, 2, 2],
        strides=[1, 2, 2, 2, 1],
        radius=0.15,
        radius_scaling=1.5,
        nsample=32,
        expansion=4,
        sa_layers=2,
        sa_use_res=True,
        normalize_dp=True,
        **kwargs,
    )
