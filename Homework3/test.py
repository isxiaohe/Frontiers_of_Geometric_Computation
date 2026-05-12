"""Evaluate a trained model checkpoint on ModelNet40 test set.

Usage:
    uv run python test.py configs/pointnet2_ssg.yaml
    uv run python test.py configs/pointnext_s.yaml --ckpt checkpoints/pointnext_s/best_model.pth
    uv run python test.py configs/pointnet2_ssg.yaml --device cpu
"""
import sys
import argparse
import yaml
from pathlib import Path

import torch
from tqdm import tqdm

from data.modelnet40 import build_dataloader
from train import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on ModelNet40")
    parser.add_argument("config", type=str, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints/{model}/best_model.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--num-points", type=int, default=None,
                        help="Override number of points")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Evaluating", ncols=100)
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        preds = logits.argmax(dim=1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix(acc=f"{correct/total:.4f}")

    acc = correct / total
    return acc, torch.cat(all_preds), torch.cat(all_labels)


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Override
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.num_points:
        config["num_points"] = args.num_points

    # Checkpoint path
    ckpt_path = args.ckpt or f"checkpoints/{config['model']}/best_model.pth"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Data (test only)
    _, test_loader = build_dataloader(config)

    # Model
    model = get_model(config).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Config:    {args.config}")
    print(f"Model:     {config['model']}")
    print(f"Params:    {num_params:,}")
    print(f"Checkpoint:{ckpt_path}")
    print(f"Device:    {device}")
    print()

    acc, preds, labels = evaluate(model, test_loader, device)

    print(f"\nTest accuracy: {acc:.4f} ({int(acc * len(labels))}/{len(labels)})")


if __name__ == "__main__":
    main()
