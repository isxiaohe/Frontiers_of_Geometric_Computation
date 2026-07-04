"""Small point-cloud denoising networks for toy target-prediction experiments."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, features: int = 32) -> None:
        super().__init__()
        if features < 2 or features % 2 != 0:
            raise ValueError("time embedding features must be an even integer >= 2")
        self.features = features

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError("t must have shape (batch,)")
        half = self.features // 2
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                math.log(1000.0),
                steps=half,
                device=t.device,
                dtype=t.dtype,
            )
        )
        phases = t[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class TinyPointDenoiser(nn.Module):
    """Pointwise MLP that predicts a 3D vector for each input point."""

    def __init__(self, hidden_dim: int = 128, time_features: int = 32) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_features)
        self.net = nn.Sequential(
            nn.Linear(3 + time_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError("x_t must have shape (batch, points, 3)")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape (batch,)")
        time = self.time_embedding(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        return self.net(torch.cat([x_t, time], dim=-1))


class PointNetContextDenoiser(nn.Module):
    """PointNet-style set-context denoiser.

    This keeps permutation equivariance: point features are encoded independently,
    pooled with a symmetric max operation, then broadcast back to each point.
    """

    def __init__(self, hidden_dim: int = 128, context_dim: int = 256, time_features: int = 32) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_features)
        point_input_dim = 3 + time_features
        self.point_encoder = nn.Sequential(
            nn.Linear(point_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(point_input_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError("x_t must have shape (batch, points, 3)")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape (batch,)")
        time = self.time_embedding(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        point_input = torch.cat([x_t, time], dim=-1)
        point_features = self.point_encoder(point_input)
        global_context = point_features.max(dim=1, keepdim=True).values.expand(-1, x_t.shape[1], -1)
        return self.decoder(torch.cat([point_input, global_context], dim=-1))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SharedMLP1d(nn.Module):
    """PVD-style shared MLP implemented as 1x1 Conv1d + GroupNorm + Swish."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            Swish(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class PVConvBlock(nn.Module):
    """Pure-PyTorch approximation of PVD's PVConv block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        resolution: int,
        dropout: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.eps = eps

        voxel_layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            Swish(),
        ]
        if dropout is not None and dropout > 0:
            voxel_layers.append(nn.Dropout(dropout))
        voxel_layers.extend(
            [
                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
                Swish(),
            ]
        )
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP1d(in_channels, out_channels)

    def voxelize(self, features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, channels = features.shape
        resolution = self.resolution

        centered = coords.detach() - coords.detach().mean(dim=1, keepdim=True)
        scale = centered.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True) * 2.0 + self.eps
        voxel_coords = torch.clamp((centered / scale + 0.5) * resolution, 0.0, resolution - 1.0)
        indices = voxel_coords.round().long()
        flat_indices = indices[..., 0] * resolution * resolution + indices[..., 1] * resolution + indices[..., 2]

        grid_flat = features.new_zeros(batch_size, resolution**3, channels)
        grid_flat.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, channels), features)

        counts = features.new_zeros(batch_size, resolution**3, 1)
        counts.scatter_add_(
            1,
            flat_indices.unsqueeze(-1),
            torch.ones(batch_size, num_points, 1, device=features.device, dtype=features.dtype),
        )
        voxel_grid = grid_flat / counts.clamp_min(1.0)
        voxel_grid = voxel_grid.permute(0, 2, 1).reshape(batch_size, channels, resolution, resolution, resolution)

        sample_coords = voxel_coords / max(resolution - 1, 1) * 2.0 - 1.0
        return voxel_grid, sample_coords

    @staticmethod
    def devoxelize(voxel_features: torch.Tensor, sample_coords: torch.Tensor) -> torch.Tensor:
        sample_grid = sample_coords[..., [2, 1, 0]].view(sample_coords.shape[0], sample_coords.shape[1], 1, 1, 3)
        sampled = F.grid_sample(
            voxel_features,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        voxel_features, sample_coords = self.voxelize(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = self.devoxelize(voxel_features, sample_coords)
        point_features = self.point_features(features.permute(0, 2, 1)).permute(0, 2, 1)
        return voxel_features + point_features


class MiniVoxelPointDenoiser(nn.Module):
    """Pure-PyTorch PVConv-inspired denoiser.

    PVD's PVCNN mixes pointwise MLP features with voxelized 3D convolutions.
    This small version keeps that inductive bias without custom CUDA kernels.
    """

    def __init__(self, hidden_dim: int = 128, time_features: int = 32, voxel_resolution: int = 12) -> None:
        super().__init__()
        if voxel_resolution < 2:
            raise ValueError("voxel_resolution must be >= 2")
        self.voxel_resolution = voxel_resolution
        self.time_embedding = SinusoidalTimeEmbedding(time_features)
        point_input_dim = 3 + time_features
        self.pvconv1 = PVConvBlock(
            point_input_dim,
            hidden_dim,
            resolution=voxel_resolution,
            dropout=None,
        )
        self.pvconv2 = PVConvBlock(
            hidden_dim + time_features,
            hidden_dim,
            resolution=voxel_resolution,
            dropout=None,
        )
        self.classifier = nn.Sequential(
            SharedMLP1d(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, 3, kernel_size=1),
        )

    def _voxelize(self, features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pvconv1.voxelize(features, coords)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError("x_t must have shape (batch, points, 3)")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape (batch,)")
        time = self.time_embedding(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        features = self.pvconv1(torch.cat([x_t, time], dim=-1), x_t)
        features = self.pvconv2(torch.cat([features, time], dim=-1), x_t)
        return self.classifier(features.permute(0, 2, 1)).permute(0, 2, 1)


class PVDLiteDenoiser(nn.Module):
    """PVConv denoiser with a lightweight global context branch.

    This is still smaller than PVD's full PVCNN2 SA/FP hierarchy, but it mirrors
    the important pattern that local point/voxel features are decoded with
    shape-level context rather than independently per point.
    """

    def __init__(self, hidden_dim: int = 128, time_features: int = 32, voxel_resolution: int = 12) -> None:
        super().__init__()
        if voxel_resolution < 2:
            raise ValueError("voxel_resolution must be >= 2")
        self.time_embedding = SinusoidalTimeEmbedding(time_features)
        point_input_dim = 3 + time_features
        self.pvconv1 = PVConvBlock(point_input_dim, hidden_dim, resolution=voxel_resolution)
        self.pvconv2 = PVConvBlock(hidden_dim + time_features, hidden_dim, resolution=voxel_resolution)
        self.global_encoder = nn.Sequential(
            SharedMLP1d(hidden_dim + time_features, hidden_dim),
            SharedMLP1d(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            SharedMLP1d(hidden_dim * 2 + time_features, hidden_dim),
            SharedMLP1d(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, 3, kernel_size=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError("x_t must have shape (batch, points, 3)")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape (batch,)")
        time = self.time_embedding(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        features = self.pvconv1(torch.cat([x_t, time], dim=-1), x_t)
        features = self.pvconv2(torch.cat([features, time], dim=-1), x_t)

        global_input = torch.cat([features, time], dim=-1).permute(0, 2, 1)
        global_context = self.global_encoder(global_input).max(dim=-1, keepdim=True).values
        global_context = global_context.expand(-1, -1, x_t.shape[1]).permute(0, 2, 1)

        decoder_input = torch.cat([features, global_context, time], dim=-1).permute(0, 2, 1)
        return self.decoder(decoder_input).permute(0, 2, 1)


def build_point_denoiser(
    model_name: str,
    *,
    hidden_dim: int = 128,
    time_features: int = 32,
    voxel_resolution: int = 12,
) -> nn.Module:
    if model_name == "tiny":
        return TinyPointDenoiser(hidden_dim=hidden_dim, time_features=time_features)
    if model_name == "pointnet":
        return PointNetContextDenoiser(
            hidden_dim=hidden_dim,
            context_dim=max(hidden_dim * 2, 32),
            time_features=time_features,
        )
    if model_name == "voxelpoint":
        return MiniVoxelPointDenoiser(
            hidden_dim=hidden_dim,
            time_features=time_features,
            voxel_resolution=voxel_resolution,
        )
    if model_name == "pvdlite":
        return PVDLiteDenoiser(
            hidden_dim=hidden_dim,
            time_features=time_features,
            voxel_resolution=voxel_resolution,
        )
    raise ValueError(f"unknown model: {model_name!r}")
