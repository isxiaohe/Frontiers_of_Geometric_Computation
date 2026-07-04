"""Schedulers for VP diffusion sampling."""

from dataclasses import dataclass
from typing import Iterator

import torch

from .targets import PredictionTarget, diffusion_coefficients, prediction_to_x0_eps, training_target


@dataclass(frozen=True)
class VPDiffusionScheduler:
    """Deterministic DDIM scheduler for the cosine VP path."""

    num_steps: int = 64
    t_min: float = 0.02
    t_max: float = 0.98

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 <= self.t_min < self.t_max <= 1.0:
            raise ValueError("require 0 <= t_min < t_max <= 1")

    def time_pairs(self, *, device: torch.device, dtype: torch.dtype) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        boundaries = torch.linspace(self.t_max, self.t_min, self.num_steps + 1, device=device, dtype=dtype)
        for index in range(self.num_steps):
            yield boundaries[index], boundaries[index + 1]

    def ddim_step(
        self,
        *,
        x_t: torch.Tensor,
        prediction: torch.Tensor,
        prediction_target: PredictionTarget,
        t: torch.Tensor,
        next_t: torch.Tensor,
    ) -> torch.Tensor:
        alpha, sigma = diffusion_coefficients(t)
        x0, epsilon = prediction_to_x0_eps(
            target=prediction_target,
            prediction=prediction,
            x_t=x_t,
            alpha=alpha,
            sigma=sigma,
        )
        next_alpha, next_sigma = diffusion_coefficients(next_t)
        return next_alpha.view(-1, 1, 1) * x0 + next_sigma.view(-1, 1, 1) * epsilon


@dataclass(frozen=True)
class DDPMDiffusionScheduler:
    """PVD-style discrete DDPM scheduler with fixed-small variance."""

    num_train_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        if self.num_train_steps <= 0:
            raise ValueError("num_train_steps must be positive")
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ValueError("require 0 < beta_start <= beta_end < 1")

        betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]])

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = posterior_variance.clamp_min(1e-20).log()
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "alphas", alphas)
        object.__setattr__(self, "alphas_cumprod", alphas_cumprod)
        object.__setattr__(self, "alphas_cumprod_prev", alphas_cumprod_prev)
        object.__setattr__(self, "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        object.__setattr__(self, "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        object.__setattr__(self, "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        object.__setattr__(self, "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        object.__setattr__(self, "posterior_variance", posterior_variance)
        object.__setattr__(self, "posterior_log_variance_clipped", posterior_log_variance_clipped)
        object.__setattr__(self, "posterior_mean_coef1", posterior_mean_coef1)
        object.__setattr__(self, "posterior_mean_coef2", posterior_mean_coef2)

    @staticmethod
    def _extract(values: torch.Tensor, t_index: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        if t_index.ndim != 1:
            raise ValueError("t_index must have shape (batch,)")
        gathered = values.to(device=t_index.device).gather(0, t_index)
        return gathered.view(t_index.shape[0], *([1] * (len(shape) - 1)))

    def model_time(self, t_index: torch.Tensor) -> torch.Tensor:
        if self.num_train_steps == 1:
            return torch.zeros_like(t_index, dtype=torch.float32)
        return t_index.float() / float(self.num_train_steps - 1)

    def sample_training_timesteps(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_steps, size=(batch_size,), device=device)

    def reverse_timesteps(self, *, device: torch.device, num_steps: int | None = None) -> Iterator[torch.Tensor]:
        if num_steps is None:
            num_steps = self.num_train_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if num_steps > self.num_train_steps:
            raise ValueError("num_steps cannot exceed num_train_steps")
        values = torch.linspace(self.num_train_steps - 1, 0, steps=num_steps, device=device)
        for value in values.round().long().unique_consecutive():
            yield value.to(device=device)

    def training_coefficients(
        self,
        t_index: torch.Tensor,
        shape: torch.Size | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._extract(self.sqrt_alphas_cumprod, t_index, shape),
            self._extract(self.sqrt_one_minus_alphas_cumprod, t_index, shape),
        )

    def q_sample(self, *, x0: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if x0.shape != noise.shape:
            raise ValueError("x0 and noise must have the same shape")
        alpha, sigma = self.training_coefficients(t_index, x0.shape)
        return alpha.to(x0.device) * x0 + sigma.to(x0.device) * noise

    def prediction_to_x0_eps(
        self,
        *,
        prediction_target: PredictionTarget,
        prediction: torch.Tensor,
        x_t: torch.Tensor,
        t_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, sigma = self.training_coefficients(t_index, x_t.shape)
        alpha = alpha.to(x_t.device)
        sigma = sigma.to(x_t.device)
        eps = torch.finfo(x_t.dtype).eps

        if prediction_target == "x0":
            x0 = prediction
            epsilon = (x_t - alpha * x0) / sigma.clamp_min(eps)
            return x0, epsilon
        if prediction_target == "epsilon":
            epsilon = prediction
            x0 = (x_t - sigma * epsilon) / alpha.clamp_min(eps)
            return x0, epsilon
        if prediction_target == "v":
            x0 = alpha * x_t - sigma * prediction
            epsilon = sigma * x_t + alpha * prediction
            return x0, epsilon
        raise ValueError(f"unknown prediction target: {prediction_target!r}")

    def training_target(
        self,
        *,
        prediction_target: PredictionTarget,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t_index: torch.Tensor,
    ) -> torch.Tensor:
        alpha, sigma = self.training_coefficients(t_index, x0.shape)
        return training_target(prediction_target, x0, noise, alpha.flatten(), sigma.flatten())

    def q_posterior_mean_variance(
        self,
        *,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            self._extract(self.posterior_mean_coef1, t_index, x_t.shape).to(x_t.device) * x0
            + self._extract(self.posterior_mean_coef2, t_index, x_t.shape).to(x_t.device) * x_t
        )
        variance = self._extract(self.posterior_variance, t_index, x_t.shape).to(x_t.device)
        log_variance = self._extract(self.posterior_log_variance_clipped, t_index, x_t.shape).to(x_t.device)
        return mean, variance, log_variance

    def p_sample_step(
        self,
        *,
        x_t: torch.Tensor,
        prediction: torch.Tensor,
        prediction_target: PredictionTarget,
        t_index: torch.Tensor,
        noise: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_x0, _ = self.prediction_to_x0_eps(
            prediction_target=prediction_target,
            prediction=prediction,
            x_t=x_t,
            t_index=t_index,
        )
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        model_mean, _, model_log_variance = self.q_posterior_mean_variance(x0=pred_x0, x_t=x_t, t_index=t_index)
        nonzero_mask = (t_index != 0).float().view(t_index.shape[0], *([1] * (x_t.ndim - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample, pred_x0
