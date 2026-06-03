"""Unified training script for PointNet++ / PointNeXt on ModelNet40.

Usage:
    uv run python train.py configs/pointnet2_ssg.yaml
    uv run python train.py configs/pointnet2_msg.yaml
    uv run python train.py configs/pointnext.yaml
"""
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import wandb

from data.modelnet40 import build_dataloader


def get_model(config):
    """Build model from config name."""
    name = config["model"]
    num_classes = config["num_classes"]
    use_normals = config.get("use_normals", False)

    if name == "pointnet2_ssg":
        from models.pointnet2 import PointNet2ClsSSG
        return PointNet2ClsSSG(num_classes=num_classes, normal_channel=use_normals)
    elif name == "pointnet2_msg":
        from models.pointnet2 import PointNet2ClsMSG
        return PointNet2ClsMSG(num_classes=num_classes, normal_channel=use_normals)
    elif name == "pointnext_s":
        from models.pointnext import pointnext_s
        return pointnext_s(num_classes=num_classes)
    elif name == "pointnext_b":
        from models.pointnext import pointnext_b
        return pointnext_b(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    grad_norm_clip=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]",
                leave=False, ncols=100)
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += points.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [test] ",
                leave=False, ncols=100)
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        loss = criterion(logits, labels)

        total_loss += loss.item() * points.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += points.size(0)

        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pointnet2_ssg.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = build_dataloader(config)

    # Model
    model = get_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']}  |  Params: {num_params:,}")

    # Optimizer
    opt_name = config.get("optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                                weight_decay=config.get("weight_decay", 1e-4))
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                              momentum=0.9,
                              weight_decay=config.get("weight_decay", 1e-4))
    elif opt_name == "muon":
        sys.path.insert(0, "submodules/Muon")
        from muon import SingleDeviceMuonWithAuxAdam
        # Muon for 2D/4D matrix weights (Linear + Conv2d); AdamW for bias, BN, Conv1d
        muon_params = [p for p in model.parameters() if p.ndim >= 2 and p.ndim != 3]
        adam_params = [p for p in model.parameters() if p.ndim < 2 or p.ndim == 3]
        param_groups = [
            dict(params=adam_params, lr=config.get("aux_lr", config["lr"] / 10),
                 weight_decay=config.get("weight_decay", 0),
                 betas=(0.9, 0.95), eps=1e-10, use_muon=False),
            dict(params=muon_params, lr=config["lr"],
                 momentum=config.get("momentum", 0.95),
                 weight_decay=config.get("weight_decay", 0), use_muon=True),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        print(f"Muon groups: {len(muon_params)} matrix params, {len(adam_params)} scalar/Conv1d params")
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=config.get("weight_decay", 1e-4))

    # Scheduler
    sched_name = config.get("scheduler", "cosine").lower()
    if sched_name == "step":
        scheduler = StepLR(optimizer,
                           step_size=config.get("step_size", 20),
                           gamma=config.get("decay_rate", 0.7))
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    grad_norm_clip = config.get("grad_norm_clip", 0)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.0))

    # Wandb (local)
    wb = config.get("wandb", {})
    wandb.init(
        project=wb.get("project", "pointnet2-modelnet40"),
        name=wb.get("name", config["model"]),
        mode=wb.get("mode", "offline"),
        config=config,
    )

    # Checkpoint dir
    ckpt_dir = Path(f"checkpoints/{config['model']}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    total_epochs = config["epochs"]
    epoch_pbar = tqdm(range(1, total_epochs + 1), desc="Training",
                      ncols=100, unit="epoch")

    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs,
            grad_norm_clip)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, epoch, total_epochs)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "lr": lr,
        }, step=epoch)

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        # Update outer epoch bar with summary
        epoch_pbar.set_postfix(
            t_loss=f"{train_loss:.4f}", t_acc=f"{train_acc:.4f}",
            v_acc=f"{test_acc:.4f}", best=f"{best_acc:.4f}",
            lr=f"{lr:.6f}", time=f"{dt:.1f}s",
        )

    print(f"\nDone. Best test acc: {best_acc:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best_model.pth'}")

    wandb.finish()


if __name__ == "__main__":
    main()
