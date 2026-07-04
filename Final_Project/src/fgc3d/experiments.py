"""Small reproducible experiments for inspecting generated point clouds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Literal

import torch

from .data import PrimitivePointCloudDataset
from .flow_matching import EulerFlowScheduler, flow_matching_condition_number, sample_flow
from .metrics import point_cloud_distribution_metrics
from .sample import sample_points
from .scheduler import DDPMDiffusionScheduler
from .targets import PredictionTarget
from .train import train_toy_model


def _point_stats(points: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(points.mean()),
        "std": float(points.std()),
        "max_abs": float(points.abs().max()),
        "rms_radius": float(points.norm(dim=-1).mean()),
    }


def _plot_point_cloud_grid(
    *,
    output_path: Path,
    reference: torch.Tensor,
    samples_by_target: dict[str, torch.Tensor],
    max_shapes_per_row: int = 4,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, torch.Tensor]] = [("reference", reference)]
    rows.extend((target, samples) for target, samples in samples_by_target.items())
    cols = min(max_shapes_per_row, min(points.shape[0] for _, points in rows))

    fig = plt.figure(figsize=(3.0 * cols, 3.0 * len(rows)))
    for row_index, (label, clouds) in enumerate(rows):
        for col_index in range(cols):
            ax = fig.add_subplot(len(rows), cols, row_index * cols + col_index + 1, projection="3d")
            cloud = clouds[col_index].detach().cpu()
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=5, alpha=0.8)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if col_index == 0:
                ax.set_ylabel(label)
            ax.set_title(f"{label} #{col_index}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_primitives_generation_experiment(
    *,
    output_dir: Path,
    targets: Iterable[PredictionTarget] = ("x0", "epsilon", "v"),
    steps: int = 1000,
    batch_size: int = 32,
    num_shapes: int = 512,
    num_points: int = 128,
    hidden_dim: int = 128,
    num_samples: int = 8,
    sample_steps: int = 64,
    seed: int = 0,
    objective: Literal["flow", "diffusion"] = "flow",
    loss_mode: Literal["target", "x0", "epsilon", "v"] = "target",
    model_name: str = "tiny",
    voxel_resolution: int = 12,
    diffusion_schedule: Literal["vp", "ddpm"] = "vp",
    ddpm_steps: int = 1000,
    device: str | torch.device = "cpu",
    write_plots: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(targets)
    samples_by_target: dict[str, torch.Tensor] = {}
    summary: dict[str, object] = {
        "dataset": "primitives",
        "steps": steps,
        "batch_size": batch_size,
        "num_shapes": num_shapes,
        "num_points": num_points,
        "hidden_dim": hidden_dim,
        "num_samples": num_samples,
        "sample_steps": sample_steps,
        "seed": seed,
        "objective": objective,
        "loss_mode": loss_mode,
        "model": model_name,
        "voxel_resolution": voxel_resolution,
        "diffusion_schedule": diffusion_schedule,
        "ddpm_steps": ddpm_steps,
        "device": None,
        "targets": {},
    }

    reference_dataset = PrimitivePointCloudDataset(num_shapes=num_samples, num_points=num_points, seed=seed + 10_000)
    reference = torch.stack([reference_dataset[index] for index in range(num_samples)])
    torch.save(reference, output_dir / "reference_samples.pt")
    summary["reference_stats"] = _point_stats(reference)

    train_seed = seed
    sample_seed = seed + 1_000
    for target in targets:
        run = train_toy_model(
            prediction_target=target,
            objective=objective,
            dataset_name="primitives",
            steps=steps,
            batch_size=batch_size,
            num_shapes=num_shapes,
            num_points=num_points,
            hidden_dim=hidden_dim,
            seed=train_seed,
            loss_mode=loss_mode,
            model_name=model_name,
            voxel_resolution=voxel_resolution,
            diffusion_schedule=diffusion_schedule,
            ddpm_steps=ddpm_steps,
            device=device,
        )
        summary["device"] = str(run.device)
        if objective == "flow":
            flow_scheduler = EulerFlowScheduler(num_steps=sample_steps)
            samples = sample_flow(
                run.model,
                scheduler=flow_scheduler,
                prediction_target=target,
                num_shapes=num_samples,
                num_points=num_points,
                seed=sample_seed,
            )
        else:
            sample_scheduler = None
            if diffusion_schedule == "ddpm":
                sample_scheduler = DDPMDiffusionScheduler(num_train_steps=ddpm_steps)
            samples = sample_points(
                run.model,
                prediction_target=target,
                num_shapes=num_samples,
                num_points=num_points,
                steps=sample_steps,
                seed=sample_seed,
                scheduler=sample_scheduler,
            )
        samples_by_target[target] = samples.detach().cpu()
        torch.save(run.model.state_dict(), output_dir / f"{target}_model.pt")
        torch.save(samples_by_target[target], output_dir / f"{target}_samples.pt")
        losses = run.metrics.losses
        target_summary = {
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "last50_mean": sum(losses[-50:]) / min(50, len(losses)),
            "sample_stats": _point_stats(samples_by_target[target]),
            "metrics": point_cloud_distribution_metrics(samples=samples_by_target[target], reference=reference),
            "train_seed": train_seed,
            "sample_seed": sample_seed,
        }
        if objective == "flow":
            target_summary["sampling_condition_number"] = flow_matching_condition_number(target, flow_scheduler)
        summary["targets"][target] = target_summary

    if write_plots:
        _plot_point_cloud_grid(
            output_path=output_dir / "generated_primitives_grid.png",
            reference=reference,
            samples_by_target=samples_by_target,
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
