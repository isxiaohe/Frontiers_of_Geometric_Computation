"""Back-to-Basics-style low-dimensional manifold toy experiment."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import SinusoidalTimeEmbedding
from .targets import PredictionTarget
from .train import StepMetrics, TrainingMetrics, train_step


class ProjectedSphereDataset(Dataset[torch.Tensor]):
    """Unit-sphere data in R^d projected into an observed R^D space."""

    def __init__(
        self,
        num_samples: int = 1024,
        intrinsic_dim: int = 2,
        ambient_dim: int = 16,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if intrinsic_dim <= 1:
            raise ValueError("intrinsic_dim must be greater than 1")
        if ambient_dim < intrinsic_dim:
            raise ValueError("ambient_dim must be >= intrinsic_dim")
        self.num_samples = num_samples
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.seed = seed

        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(ambient_dim, intrinsic_dim, generator=generator)
        q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.projection = q.contiguous()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        latent = torch.randn(self.intrinsic_dim, generator=generator)
        latent = latent / latent.norm().clamp_min(1e-8)
        return latent @ self.projection.T


class VectorDenoiser(nn.Module):
    """Five-layer MLP denoiser for D-dimensional vectors."""

    def __init__(self, ambient_dim: int, hidden_dim: int = 256, time_features: int = 32) -> None:
        super().__init__()
        if ambient_dim <= 0:
            raise ValueError("ambient_dim must be positive")
        self.ambient_dim = ambient_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_features)
        self.net = nn.Sequential(
            nn.Linear(ambient_dim + time_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ambient_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[-1] != self.ambient_dim:
            raise ValueError(f"x_t must have shape (batch, {self.ambient_dim})")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape (batch,)")
        time = self.time_embedding(t)
        return self.net(torch.cat([x_t, time], dim=-1))


@dataclass(frozen=True)
class ProjectedSphereRun:
    prediction_target: str
    losses: list[float]


def run_projected_sphere_training(
    *,
    prediction_target: PredictionTarget,
    steps: int = 200,
    batch_size: int = 64,
    num_samples: int = 1024,
    intrinsic_dim: int = 2,
    ambient_dim: int = 16,
    hidden_dim: int = 256,
    seed: int = 0,
    lr: float = 1e-3,
    loss_mode: str = "v",
) -> TrainingMetrics:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    dataset = ProjectedSphereDataset(
        num_samples=num_samples,
        intrinsic_dim=intrinsic_dim,
        ambient_dim=ambient_dim,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VectorDenoiser(ambient_dim=ambient_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics: StepMetrics = train_step(
            model,
            optimizer,
            batch,
            prediction_target=prediction_target,
            loss_mode=loss_mode,
        )
        losses.append(metrics.loss)
    return TrainingMetrics(prediction_target=prediction_target, losses=losses)
