"""Train one ShapeNet category model and save generated point clouds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fgc3d.data import ShapeNetPointCloudDataset
from fgc3d.flow_matching import EulerFlowScheduler, flow_matching_condition_number, sample_flow
from fgc3d.sample import sample_points
from fgc3d.scheduler import DDPMDiffusionScheduler
from fgc3d.train import train_toy_model


def _point_stats(points: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(points.mean()),
        "std": float(points.std()),
        "max_abs": float(points.abs().max()),
        "rms_radius": float(points.norm(dim=-1).mean()),
    }


def _plot_grid(*, output_path: Path, reference: torch.Tensor, samples: torch.Tensor, label: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("reference", reference), (label, samples)]
    cols = min(4, reference.shape[0], samples.shape[0])
    fig = plt.figure(figsize=(3.0 * cols, 6.0))
    for row_index, (row_label, clouds) in enumerate(rows):
        for col_index in range(cols):
            ax = fig.add_subplot(2, cols, row_index * cols + col_index + 1, projection="3d")
            cloud = clouds[col_index].detach().cpu()
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=4, alpha=0.8)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"{row_label} #{col_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shapenet_category_debug"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", default="chair")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--objective", choices=["flow", "diffusion"], default="flow")
    parser.add_argument("--prediction-target", choices=["x0", "epsilon", "v"], default="x0")
    parser.add_argument("--loss-mode", choices=["target", "x0", "epsilon", "v"], default="target")
    parser.add_argument("--model", choices=["tiny", "pointnet", "voxelpoint", "pvdlite"], default="pvdlite")
    parser.add_argument("--diffusion-schedule", choices=["vp", "ddpm"], default="ddpm")
    parser.add_argument("--ddpm-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--voxel-resolution", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="Training/sampling device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = train_toy_model(
        prediction_target=args.prediction_target,
        steps=args.steps,
        batch_size=args.batch_size,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        lr=args.lr,
        loss_mode=args.loss_mode,
        dataset_name="shapenet",
        model_name=args.model,
        voxel_resolution=args.voxel_resolution,
        data_root=str(args.data_root),
        category=args.category,
        split=args.split,
        objective=args.objective,
        diffusion_schedule=args.diffusion_schedule,
        ddpm_steps=args.ddpm_steps,
        device=args.device,
    )
    if args.objective == "flow":
        flow_scheduler = EulerFlowScheduler(num_steps=args.sample_steps)
        samples = sample_flow(
            run.model,
            scheduler=flow_scheduler,
            prediction_target=args.prediction_target,
            num_shapes=args.num_samples,
            num_points=args.num_points,
            seed=args.seed + 1000,
        ).detach().cpu()
    else:
        scheduler = None
        if args.diffusion_schedule == "ddpm":
            scheduler = DDPMDiffusionScheduler(num_train_steps=args.ddpm_steps)
        samples = sample_points(
            run.model,
            prediction_target=args.prediction_target,
            num_shapes=args.num_samples,
            num_points=args.num_points,
            steps=args.sample_steps,
            seed=args.seed + 1000,
            scheduler=scheduler,
        ).detach().cpu()
    reference_dataset = ShapeNetPointCloudDataset(
        root_dir=args.data_root,
        category=args.category,
        split=args.split,
        num_points=args.num_points,
        seed=args.seed + 2000,
    )
    reference_count = min(args.num_samples, len(reference_dataset))
    reference = torch.stack([reference_dataset[index] for index in range(reference_count)])

    torch.save(run.model.state_dict(), args.output_dir / f"{args.prediction_target}_model.pt")
    torch.save(samples, args.output_dir / f"{args.prediction_target}_samples.pt")
    torch.save(reference, args.output_dir / "reference_samples.pt")

    summary = {
        "dataset": "shapenet",
        "data_root": str(args.data_root),
        "category": args.category,
        "split": args.split,
        "objective": args.objective,
        "prediction_target": args.prediction_target,
        "loss_mode": args.loss_mode,
        "model": args.model,
        "diffusion_schedule": args.diffusion_schedule,
        "ddpm_steps": args.ddpm_steps,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_points": args.num_points,
        "hidden_dim": args.hidden_dim,
        "voxel_resolution": args.voxel_resolution,
        "device": str(run.device),
        "first_loss": run.metrics.losses[0],
        "final_loss": run.metrics.losses[-1],
        "last50_mean": sum(run.metrics.losses[-50:]) / min(50, len(run.metrics.losses)),
        "sample_stats": _point_stats(samples),
        "reference_stats": _point_stats(reference),
    }
    if args.objective == "flow":
        summary["sampling_condition_number"] = flow_matching_condition_number(
            args.prediction_target,
            flow_scheduler,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        _plot_grid(
            output_path=args.output_dir / "generated_shapenet_grid.png",
            reference=reference,
            samples=samples,
            label=args.prediction_target,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
