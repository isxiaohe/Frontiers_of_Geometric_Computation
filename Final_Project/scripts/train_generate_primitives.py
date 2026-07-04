"""Train x0/epsilon/v primitive point-cloud models and save generated samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fgc3d.experiments import run_primitives_generation_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/primitives_debug"))
    parser.add_argument("--targets", nargs="+", choices=["x0", "epsilon", "v"], default=["x0", "epsilon", "v"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-shapes", type=int, default=512)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", choices=["flow", "diffusion"], default="flow")
    parser.add_argument("--loss-mode", choices=["target", "x0", "epsilon", "v"], default="target")
    parser.add_argument("--model", choices=["tiny", "pointnet", "voxelpoint", "pvdlite"], default="tiny")
    parser.add_argument("--voxel-resolution", type=int, default=12)
    parser.add_argument("--diffusion-schedule", choices=["vp", "ddpm"], default="vp")
    parser.add_argument("--ddpm-steps", type=int, default=1000)
    parser.add_argument("--device", default="auto", help="Training/sampling device: auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_primitives_generation_experiment(
        output_dir=args.output_dir,
        targets=tuple(args.targets),
        steps=args.steps,
        batch_size=args.batch_size,
        num_shapes=args.num_shapes,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        num_samples=args.num_samples,
        sample_steps=args.sample_steps,
        seed=args.seed,
        objective=args.objective,
        loss_mode=args.loss_mode,
        model_name=args.model,
        voxel_resolution=args.voxel_resolution,
        diffusion_schedule=args.diffusion_schedule,
        ddpm_steps=args.ddpm_steps,
        device=args.device,
        write_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
