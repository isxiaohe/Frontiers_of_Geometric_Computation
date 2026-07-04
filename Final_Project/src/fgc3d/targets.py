"""Prediction target parameterizations for a small flow/diffusion baseline."""

from typing import Literal

import torch

PredictionTarget = Literal["x0", "epsilon", "v"]
LossTarget = Literal["target", "x0", "epsilon", "v"]


def _broadcast_coefficients(coefficients: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if coefficients.ndim != 1:
        raise ValueError("diffusion coefficients must have shape (batch,)")
    if coefficients.shape[0] != points.shape[0]:
        raise ValueError("coefficient batch size must match point-cloud batch size")
    return coefficients.view(coefficients.shape[0], *([1] * (points.ndim - 1)))


def diffusion_coefficients(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map normalized times in [0, 1] to alpha/sigma coefficients."""
    if t.ndim != 1:
        raise ValueError("t must have shape (batch,)")
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("t must stay in [0, 1]")
    angles = t * (torch.pi / 2.0)
    return torch.cos(angles), torch.sin(angles)


def make_noisy_points(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Construct x_t = alpha * x0 + sigma * epsilon."""
    if x0.shape != epsilon.shape:
        raise ValueError("x0 and epsilon must have the same shape")
    alpha_b = _broadcast_coefficients(alpha, x0)
    sigma_b = _broadcast_coefficients(sigma, x0)
    return alpha_b * x0 + sigma_b * epsilon


def training_target(
    target: PredictionTarget,
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Return the supervised target for x0, epsilon, or v prediction."""
    if target == "x0":
        return x0
    if target == "epsilon":
        return epsilon
    if target == "v":
        alpha_b = _broadcast_coefficients(alpha, x0)
        sigma_b = _broadcast_coefficients(sigma, x0)
        return alpha_b * epsilon - sigma_b * x0
    raise ValueError(f"unknown prediction target: {target!r}")


def prediction_to_x0_eps(
    *,
    target: PredictionTarget,
    prediction: torch.Tensor,
    x_t: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a model prediction back to implied clean points and noise."""
    alpha_b = _broadcast_coefficients(alpha, x_t)
    sigma_b = _broadcast_coefficients(sigma, x_t)
    eps = torch.finfo(x_t.dtype).eps

    if target == "x0":
        x0 = prediction
        epsilon = (x_t - alpha_b * x0) / sigma_b.clamp_min(eps)
        return x0, epsilon
    if target == "epsilon":
        epsilon = prediction
        x0 = (x_t - sigma_b * epsilon) / alpha_b.clamp_min(eps)
        return x0, epsilon
    if target == "v":
        x0 = alpha_b * x_t - sigma_b * prediction
        epsilon = sigma_b * x_t + alpha_b * prediction
        return x0, epsilon
    raise ValueError(f"unknown prediction target: {target!r}")


def prediction_to_v(
    *,
    target: PredictionTarget,
    prediction: torch.Tensor,
    x_t: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Convert an x0/epsilon/v prediction to its implied velocity."""
    return prediction_to_target(
        target=target,
        prediction=prediction,
        loss_target="v",
        x_t=x_t,
        alpha=alpha,
        sigma=sigma,
    )


def prediction_to_target(
    *,
    target: PredictionTarget,
    prediction: torch.Tensor,
    loss_target: PredictionTarget,
    x_t: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Convert any prediction parameterization to x0, epsilon, or v space."""
    if loss_target == target:
        return prediction
    x0, epsilon = prediction_to_x0_eps(
        target=target,
        prediction=prediction,
        x_t=x_t,
        alpha=alpha,
        sigma=sigma,
    )
    if loss_target == "x0":
        return x0
    if loss_target == "epsilon":
        return epsilon
    if loss_target == "v":
        return training_target("v", x0, epsilon, alpha, sigma)
    raise ValueError(f"unknown loss target: {loss_target!r}")
