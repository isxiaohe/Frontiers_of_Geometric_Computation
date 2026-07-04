"""Optimal-transport flow matching path and sampler."""

from dataclasses import dataclass

import torch

from .model import TinyPointDenoiser
from .targets import PredictionTarget


def _broadcast_times(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError("t must have shape (batch,)")
    if t.shape[0] != x.shape[0]:
        raise ValueError("time batch size must match data batch size")
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def flow_matching_path(x0: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """OT/linear path x_t = (1 - t) z + t x0."""
    if x0.shape != z.shape:
        raise ValueError("x0 and z must have the same shape")
    t_b = _broadcast_times(t, x0)
    return (1.0 - t_b) * z + t_b * x0


def flow_matching_target(x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Constant velocity for the linear path from z to x0."""
    if x0.shape != z.shape:
        raise ValueError("x0 and z must have the same shape")
    return x0 - z


def flow_matching_training_target(
    target: PredictionTarget,
    x0: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Return x0, source-noise, or OT velocity supervision for the linear path."""
    _broadcast_times(t, x0)
    if target == "x0":
        return x0
    if target == "epsilon":
        return z
    if target == "v":
        return flow_matching_target(x0, z)
    raise ValueError(f"unknown prediction target: {target!r}")


def flow_matching_prediction_to_components(
    *,
    target: PredictionTarget,
    prediction: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an FM prediction to implied x0, source noise z, and velocity."""
    if prediction.shape != x_t.shape:
        raise ValueError("prediction and x_t must have the same shape")
    t_b = _broadcast_times(t, x_t)
    eps = torch.finfo(x_t.dtype).eps

    if target == "x0":
        x0 = prediction
        z = (x_t - t_b * x0) / (1.0 - t_b).clamp_min(eps)
        velocity = x0 - z
        return x0, z, velocity
    if target == "epsilon":
        z = prediction
        x0 = (x_t - (1.0 - t_b) * z) / t_b.clamp_min(eps)
        velocity = x0 - z
        return x0, z, velocity
    if target == "v":
        velocity = prediction
        z = x_t - t_b * velocity
        x0 = x_t + (1.0 - t_b) * velocity
        return x0, z, velocity
    raise ValueError(f"unknown prediction target: {target!r}")


def flow_matching_prediction_to_target(
    *,
    target: PredictionTarget,
    prediction: torch.Tensor,
    loss_target: PredictionTarget,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Convert any FM prediction parameterization into the requested loss space."""
    if loss_target == target:
        return prediction
    x0, z, velocity = flow_matching_prediction_to_components(
        target=target,
        prediction=prediction,
        x_t=x_t,
        t=t,
    )
    if loss_target == "x0":
        return x0
    if loss_target == "epsilon":
        return z
    if loss_target == "v":
        return velocity
    raise ValueError(f"unknown loss target: {loss_target!r}")


@dataclass(frozen=True)
class EulerFlowScheduler:
    """Fixed-step Euler scheduler from t=0 noise to t=1 data."""

    num_steps: int = 32

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

    @property
    def dt(self) -> float:
        return 1.0 / self.num_steps

    def times(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        indices = torch.arange(self.num_steps, device=device, dtype=dtype)
        return (indices + 0.5) / float(self.num_steps)


def flow_matching_condition_number(target: PredictionTarget, scheduler: EulerFlowScheduler) -> float:
    """Worst scalar amplification when converting a prediction to velocity."""
    if target == "v":
        return 1.0
    times = scheduler.times(device=torch.device("cpu"), dtype=torch.float64)
    if target == "epsilon":
        return float((1.0 / times).max())
    if target == "x0":
        return float((1.0 / (1.0 - times)).max())
    raise ValueError(f"unknown prediction target: {target!r}")


@torch.no_grad()
def sample_flow(
    model: TinyPointDenoiser,
    *,
    scheduler: EulerFlowScheduler,
    prediction_target: PredictionTarget = "v",
    num_shapes: int = 4,
    num_points: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    if num_shapes <= 0:
        raise ValueError("num_shapes must be positive")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn(num_shapes, num_points, 3, generator=generator, device=device, dtype=dtype)

    for t_value in scheduler.times(device=device, dtype=dtype):
        t = torch.full((num_shapes,), float(t_value), device=device, dtype=dtype)
        prediction = model(x_t, t)
        velocity = flow_matching_prediction_to_target(
            target=prediction_target,
            prediction=prediction,
            loss_target="v",
            x_t=x_t,
            t=t,
        )
        x_t = x_t + scheduler.dt * velocity

    return x_t
