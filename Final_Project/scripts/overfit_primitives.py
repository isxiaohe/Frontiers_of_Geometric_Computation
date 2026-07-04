"""Run a fixed-batch primitive overfitting diagnostic."""

from __future__ import annotations

import argparse
import json

from fgc3d.overfit import run_fixed_batch_overfit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-target", choices=["x0", "epsilon", "v"], default="x0")
    parser.add_argument("--objective", choices=["flow", "diffusion"], default="flow")
    parser.add_argument("--loss-mode", choices=["target", "x0", "epsilon", "v"], default="target")
    parser.add_argument("--model", choices=["tiny", "pointnet", "voxelpoint", "pvdlite"], default="voxelpoint")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--voxel-resolution", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_fixed_batch_overfit(
        prediction_target=args.prediction_target,
        objective=args.objective,
        loss_mode=args.loss_mode,
        model_name=args.model,
        steps=args.steps,
        batch_size=args.batch_size,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        voxel_resolution=args.voxel_resolution,
        lr=args.lr,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
