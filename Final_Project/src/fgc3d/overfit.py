"""Small fixed-batch overfitting diagnostics."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F

from .data import PrimitivePointCloudDataset
from .flow_matching import flow_matching_path, flow_matching_prediction_to_target, flow_matching_training_target
from .model import build_point_denoiser
from .targets import PredictionTarget, make_noisy_points, prediction_to_target, training_target


class OverfitSummary(TypedDict):
    prediction_target: str
    objective: str
    loss_mode: str
    model: str
    first_loss: float
    final_loss: float
    loss_reduction: float
    losses: list[float]


def _fixed_times(batch_size: int, *, device: torch.device) -> torch.Tensor:
    if batch_size == 1:
        return torch.tensor([0.5], device=device)
    return torch.linspace(0.2, 0.8, steps=batch_size, device=device)


def run_fixed_batch_overfit(
    *,
    prediction_target: PredictionTarget = "x0",
    objective: str = "flow",
    loss_mode: str = "target",
    model_name: str = "voxelpoint",
    steps: int = 200,
    batch_size: int = 2,
    num_points: int = 64,
    hidden_dim: int = 64,
    voxel_resolution: int = 6,
    lr: float = 1e-3,
    seed: int = 0,
) -> OverfitSummary:
    """Train on one fixed primitive batch, fixed t, and fixed noise."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    dataset = PrimitivePointCloudDataset(num_shapes=batch_size, num_points=num_points, seed=seed)
    x0 = torch.stack([dataset[index] for index in range(batch_size)])
    model = build_point_denoiser(model_name, hidden_dim=hidden_dim, voxel_resolution=voxel_resolution)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(seed + 10_000)
    epsilon = torch.randn(x0.shape, generator=generator)
    t = _fixed_times(batch_size, device=x0.device)
    if objective == "flow":
        x_t = flow_matching_path(x0, epsilon, t)
        expected_prediction = flow_matching_training_target(prediction_target, x0, epsilon, t)
    elif objective == "diffusion":
        alpha = torch.cos(t * (torch.pi / 2.0))
        sigma = torch.sin(t * (torch.pi / 2.0))
        x_t = make_noisy_points(x0, epsilon, alpha, sigma)
        expected_prediction = training_target(prediction_target, x0, epsilon, alpha, sigma)
    else:
        raise ValueError(f"unknown objective: {objective!r}")

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_t, t)
        if loss_mode == "target":
            loss = F.mse_loss(prediction, expected_prediction)
        elif loss_mode in {"x0", "epsilon", "v"}:
            if objective == "flow":
                expected_loss_target = flow_matching_training_target(loss_mode, x0, epsilon, t)
                predicted_loss_target = flow_matching_prediction_to_target(
                    target=prediction_target,
                    prediction=prediction,
                    loss_target=loss_mode,
                    x_t=x_t,
                    t=t,
                )
            else:
                expected_loss_target = training_target(loss_mode, x0, epsilon, alpha, sigma)
                predicted_loss_target = prediction_to_target(
                    target=prediction_target,
                    prediction=prediction,
                    loss_target=loss_mode,
                    x_t=x_t,
                    alpha=alpha,
                    sigma=sigma,
                )
            loss = F.mse_loss(predicted_loss_target, expected_loss_target)
        else:
            raise ValueError(f"unknown loss mode: {loss_mode!r}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "prediction_target": prediction_target,
        "objective": objective,
        "loss_mode": loss_mode,
        "model": model_name,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1],
        "losses": losses,
    }
