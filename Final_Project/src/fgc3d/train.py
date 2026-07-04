"""Training helpers for the tiny point-cloud baseline."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    PrimitivePointCloudDataset,
    ShapeNetPointCloudDataset,
    StructuralObjectPointCloudDataset,
    SyntheticPointCloudDataset,
)
from .device import resolve_device
from .flow_matching import flow_matching_path, flow_matching_prediction_to_target, flow_matching_training_target
from .model import build_point_denoiser
from .scheduler import DDPMDiffusionScheduler
from .targets import (
    PredictionTarget,
    diffusion_coefficients,
    make_noisy_points,
    prediction_to_target,
    training_target,
)


@dataclass(frozen=True)
class StepMetrics:
    loss: float


@dataclass(frozen=True)
class TrainingMetrics:
    prediction_target: str
    losses: list[float]


@dataclass(frozen=True)
class TrainingRun:
    model: nn.Module
    metrics: TrainingMetrics
    device: torch.device


def sample_times(batch_size: int, *, device: torch.device) -> torch.Tensor:
    """Sample interior times to avoid singular x0/epsilon conversions at endpoints."""
    return torch.rand(batch_size, device=device) * 0.96 + 0.02


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    *,
    prediction_target: PredictionTarget,
    loss_mode: str = "target",
    diffusion_scheduler: DDPMDiffusionScheduler | None = None,
) -> StepMetrics:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x0 = x0.to(next(model.parameters()).device)
    epsilon = torch.randn_like(x0)
    if diffusion_scheduler is None:
        t = sample_times(x0.shape[0], device=x0.device)
        alpha, sigma = diffusion_coefficients(t)
        x_t = make_noisy_points(x0, epsilon, alpha, sigma)
    else:
        t_index = diffusion_scheduler.sample_training_timesteps(x0.shape[0], device=x0.device)
        t = diffusion_scheduler.model_time(t_index)
        x_t = diffusion_scheduler.q_sample(x0=x0, t_index=t_index, noise=epsilon)
        alpha_b, sigma_b = diffusion_scheduler.training_coefficients(t_index, x0.shape)
        alpha, sigma = alpha_b.flatten(), sigma_b.flatten()
    expected = training_target(prediction_target, x0, epsilon, alpha, sigma)

    prediction = model(x_t, t)
    if loss_mode == "target":
        loss = F.mse_loss(prediction, expected)
    elif loss_mode in {"x0", "epsilon", "v"}:
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
    return StepMetrics(loss=float(loss.detach().cpu()))


def train_flow_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    *,
    prediction_target: PredictionTarget = "v",
    loss_mode: str = "target",
) -> StepMetrics:
    """One optimal-transport flow matching update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    x0 = x0.to(next(model.parameters()).device)
    t = torch.rand(x0.shape[0], device=x0.device)
    z = torch.randn_like(x0)
    x_t = flow_matching_path(x0, z, t)
    expected = flow_matching_training_target(prediction_target, x0, z, t)

    prediction = model(x_t, t)
    if loss_mode == "target":
        loss = F.mse_loss(prediction, expected)
    elif loss_mode in {"x0", "epsilon", "v"}:
        expected_loss_target = flow_matching_training_target(loss_mode, x0, z, t)
        predicted_loss_target = flow_matching_prediction_to_target(
            target=prediction_target,
            prediction=prediction,
            loss_target=loss_mode,
            x_t=x_t,
            t=t,
        )
        loss = F.mse_loss(predicted_loss_target, expected_loss_target)
    else:
        raise ValueError(f"unknown loss mode: {loss_mode!r}")
    loss.backward()
    optimizer.step()
    return StepMetrics(loss=float(loss.detach().cpu()))


def run_toy_training(
    *,
    prediction_target: PredictionTarget,
    steps: int = 20,
    batch_size: int = 16,
    num_shapes: int = 128,
    num_points: int = 64,
    hidden_dim: int = 128,
    seed: int = 0,
    lr: float = 1e-3,
    loss_mode: str = "target",
    dataset_name: str = "chair",
    objective: str = "flow",
    model_name: str = "tiny",
    voxel_resolution: int = 12,
    data_root: str | None = None,
    category: str = "chair",
    split: str = "train",
    diffusion_schedule: str = "vp",
    ddpm_steps: int = 1000,
    device: str | torch.device = "cpu",
) -> TrainingMetrics:
    if steps <= 0:
        raise ValueError("steps must be positive")
    run = train_toy_model(
        prediction_target=prediction_target,
        steps=steps,
        batch_size=batch_size,
        num_shapes=num_shapes,
        num_points=num_points,
        hidden_dim=hidden_dim,
        seed=seed,
        lr=lr,
        loss_mode=loss_mode,
        dataset_name=dataset_name,
        objective=objective,
        model_name=model_name,
        voxel_resolution=voxel_resolution,
        data_root=data_root,
        category=category,
        split=split,
        diffusion_schedule=diffusion_schedule,
        ddpm_steps=ddpm_steps,
        device=device,
    )
    return run.metrics


def train_toy_model(
    *,
    prediction_target: PredictionTarget,
    steps: int = 20,
    batch_size: int = 16,
    num_shapes: int = 128,
    num_points: int = 64,
    hidden_dim: int = 128,
    seed: int = 0,
    lr: float = 1e-3,
    loss_mode: str = "target",
    dataset_name: str = "chair",
    objective: str = "flow",
    model_name: str = "tiny",
    voxel_resolution: int = 12,
    data_root: str | None = None,
    category: str = "chair",
    split: str = "train",
    diffusion_schedule: str = "vp",
    ddpm_steps: int = 1000,
    device: str | torch.device = "cpu",
) -> TrainingRun:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    if dataset_name == "chair":
        dataset = SyntheticPointCloudDataset(num_shapes=num_shapes, num_points=num_points, seed=seed)
    elif dataset_name == "primitives":
        dataset = PrimitivePointCloudDataset(num_shapes=num_shapes, num_points=num_points, seed=seed)
    elif dataset_name == "structural":
        dataset = StructuralObjectPointCloudDataset(num_shapes=num_shapes, num_points=num_points, seed=seed)
    elif dataset_name == "shapenet":
        if data_root is None:
            raise ValueError("data_root is required when dataset_name='shapenet'")
        dataset = ShapeNetPointCloudDataset(
            root_dir=data_root,
            category=category,
            split=split,
            num_points=num_points,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name!r}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch_device = resolve_device(device)
    model = build_point_denoiser(model_name, hidden_dim=hidden_dim, voxel_resolution=voxel_resolution).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if diffusion_schedule == "vp":
        diffusion_scheduler = None
    elif diffusion_schedule == "ddpm":
        diffusion_scheduler = DDPMDiffusionScheduler(num_train_steps=ddpm_steps)
    else:
        raise ValueError(f"unknown diffusion schedule: {diffusion_schedule!r}")

    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if objective == "diffusion":
            metrics = train_step(
                model,
                optimizer,
                batch,
                prediction_target=prediction_target,
                loss_mode=loss_mode,
                diffusion_scheduler=diffusion_scheduler,
            )
        elif objective == "flow":
            metrics = train_flow_step(
                model,
                optimizer,
                batch,
                prediction_target=prediction_target,
                loss_mode=loss_mode,
            )
        else:
            raise ValueError(f"unknown objective: {objective!r}")
        losses.append(metrics.loss)
    metrics = TrainingMetrics(prediction_target=prediction_target, losses=losses)
    return TrainingRun(model=model, metrics=metrics, device=torch_device)
