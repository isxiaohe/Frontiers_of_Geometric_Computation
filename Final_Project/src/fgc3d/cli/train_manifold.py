"""Train the projected unit-sphere toy model."""

import argparse
import json

from fgc3d.manifold import run_projected_sphere_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-target", choices=["x0", "epsilon", "v"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--intrinsic-dim", type=int, default=2)
    parser.add_argument("--ambient-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-mode", choices=["target", "v"], default="v")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_projected_sphere_training(
        prediction_target=args.prediction_target,
        steps=args.steps,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        intrinsic_dim=args.intrinsic_dim,
        ambient_dim=args.ambient_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        lr=args.lr,
        loss_mode=args.loss_mode,
    )
    losses = metrics.losses
    print(
        json.dumps(
            {
                "prediction_target": metrics.prediction_target,
                "intrinsic_dim": args.intrinsic_dim,
                "ambient_dim": args.ambient_dim,
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "last20_mean": sum(losses[-20:]) / min(20, len(losses)),
                "loss_mode": args.loss_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
