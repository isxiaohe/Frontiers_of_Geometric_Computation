"""Simple deterministic sampler for smoke-testing trained denoisers."""

import torch
from torch import nn

from .scheduler import DDPMDiffusionScheduler, VPDiffusionScheduler
from .targets import PredictionTarget


@torch.no_grad()
def sample_points(
    model: nn.Module,
    *,
    prediction_target: PredictionTarget,
    num_shapes: int = 4,
    num_points: int = 64,
    steps: int = 8,
    seed: int = 0,
    scheduler: VPDiffusionScheduler | DDPMDiffusionScheduler | None = None,
) -> torch.Tensor:
    """Generate point clouds with a deterministic DDIM-style update."""
    if num_shapes <= 0:
        raise ValueError("num_shapes must be positive")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn(num_shapes, num_points, 3, generator=generator, device=device)
    scheduler = scheduler or VPDiffusionScheduler(num_steps=steps)

    if isinstance(scheduler, DDPMDiffusionScheduler):
        for scalar_t_index in scheduler.reverse_timesteps(device=device, num_steps=steps):
            t_index = scalar_t_index.expand(num_shapes)
            t = scheduler.model_time(t_index)
            pred = model(x_t, t)
            noise = torch.randn(x_t.shape, generator=generator, device=device)
            x_t, _ = scheduler.p_sample_step(
                x_t=x_t,
                prediction=pred,
                prediction_target=prediction_target,
                t_index=t_index,
                noise=noise,
                clip_denoised=True,
            )
        return x_t

    for t_value, next_t_value in scheduler.time_pairs(device=device, dtype=x_t.dtype):
        t = t_value.expand(num_shapes)
        next_t = next_t_value.expand(num_shapes)
        pred = model(x_t, t)
        x_t = scheduler.ddim_step(
            x_t=x_t,
            prediction=pred,
            prediction_target=prediction_target,
            t=t,
            next_t=next_t,
        )

    return x_t
