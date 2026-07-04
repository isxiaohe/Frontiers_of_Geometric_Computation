"""Train the tiny synthetic point-cloud model for a few iterations."""

import argparse
import json

from fgc3d.train import train_toy_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-target", choices=["x0", "epsilon", "v"], required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-shapes", type=int, default=128)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", choices=["chair", "primitives", "structural", "shapenet"], default="chair")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--category", default="chair")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--loss-mode", choices=["target", "x0", "epsilon", "v"], default="target")
    parser.add_argument("--objective", choices=["flow", "diffusion"], default="flow")
    parser.add_argument("--diffusion-schedule", choices=["vp", "ddpm"], default="vp")
    parser.add_argument("--ddpm-steps", type=int, default=1000)
    parser.add_argument("--model", choices=["tiny", "pointnet", "voxelpoint", "pvdlite"], default="tiny")
    parser.add_argument("--voxel-resolution", type=int, default=12)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, or mps.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = train_toy_model(
        prediction_target=args.prediction_target,
        steps=args.steps,
        batch_size=args.batch_size,
        num_shapes=args.num_shapes,
        num_points=args.num_points,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        lr=args.lr,
        loss_mode=args.loss_mode,
        dataset_name=args.dataset,
        objective=args.objective,
        model_name=args.model,
        voxel_resolution=args.voxel_resolution,
        data_root=args.data_root,
        category=args.category,
        split=args.split,
        diffusion_schedule=args.diffusion_schedule,
        ddpm_steps=args.ddpm_steps,
        device=args.device,
    )
    metrics = run.metrics
    print(
        json.dumps(
            {
                "prediction_target": metrics.prediction_target,
                "dataset": args.dataset,
                "data_root": args.data_root,
                "category": args.category,
                "split": args.split,
                "loss_mode": args.loss_mode,
                "objective": args.objective,
                "diffusion_schedule": args.diffusion_schedule,
                "ddpm_steps": args.ddpm_steps,
                "model": args.model,
                "voxel_resolution": args.voxel_resolution,
                "device": str(run.device),
                "losses": metrics.losses,
                "final_loss": metrics.losses[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
