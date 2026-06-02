import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SDFMLP, FourierFeatureMLP
from dataset import SDFDataset


def compute_gradient(sdf_pred, points):
    """通过torch.autograd.grad计算sdf_pred对points的梯度。"""
    grad_pred = torch.autograd.grad(
        outputs=sdf_pred,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad_pred


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(args.data_root, args.uid)
    dataset = SDFDataset(data_dir, use_surface_points=args.use_surface_points)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    if args.mode == "base":
        model = SDFMLP(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=args.activation,
            skip_layers=args.skip_layers,
            geometric_init=args.geometric_init,
            radius_init=args.radius_init,
        ).to(device)
    elif args.mode == "fourier":
        model = FourierFeatureMLP(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=args.activation,
            skip_layers=args.skip_layers,
            mapping_size=args.mapping_size,
            sigma=args.sigma,
            geometric_init=args.geometric_init,
            radius_init=args.radius_init,
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Model: {args.mode}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{args.mode}_log.json")
    log = []

    model.train()
    start_time = time.time()

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", ncols=100)
    for epoch in pbar:
        epoch_loss = 0.0
        epoch_sdf_loss = 0.0
        epoch_grad_loss = 0.0
        epoch_eikonal_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, ncols=100):
            points = batch["points"].to(device)
            grad_gt = batch["grad"].to(device)
            sdf_gt = batch["sdf"].to(device)

            points.requires_grad_(True)

            optimizer.zero_grad()
            sdf_pred = model(points)
            grad_pred = compute_gradient(sdf_pred, points)

            loss_sdf = nn.functional.mse_loss(sdf_pred, sdf_gt)
            loss_grad = nn.functional.mse_loss(grad_pred, grad_gt)
            loss = args.lambda_sdf * loss_sdf + args.lambda_grad * loss_grad

            # Eikonal loss: 额外随机采样点约束 |∇F| = 1
            if args.lambda_eikonal > 0:
                # 在数据空间 [-0.5, 0.5]^3 内均匀采样随机点
                empty_points = torch.rand(
                    points.shape[0] // 2, 3, device=device, dtype=points.dtype
                ) - 0.5
                empty_points.requires_grad_(True)
                empty_sdf = model(empty_points)
                empty_grad = compute_gradient(empty_sdf, empty_points)
                loss_eikonal = torch.mean(
                    (torch.norm(empty_grad, dim=-1) - 1.0) ** 2
                )
                loss = loss + args.lambda_eikonal * loss_eikonal
            else:
                loss_eikonal = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sdf_loss += loss_sdf.item()
            epoch_grad_loss += loss_grad.item()
            epoch_eikonal_loss += loss_eikonal.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_sdf_loss = epoch_sdf_loss / num_batches
        avg_grad_loss = epoch_grad_loss / num_batches
        avg_eikonal_loss = epoch_eikonal_loss / num_batches

        log.append({
            "epoch": epoch,
            "loss": avg_loss,
            "sdf_loss": avg_sdf_loss,
            "grad_loss": avg_grad_loss,
            "eikonal_loss": avg_eikonal_loss,
            "lr": scheduler.get_last_lr()[0],
        })

        postfix = {
            "loss": f"{avg_loss:.6f}",
            "sdf": f"{avg_sdf_loss:.6f}",
            "grad": f"{avg_grad_loss:.6f}",
        }
        if args.lambda_eikonal > 0:
            postfix["eikonal"] = f"{avg_eikonal_loss:.6f}"
        postfix["lr"] = f"{scheduler.get_last_lr()[0]:.6f}"
        pbar.set_postfix(postfix)

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss: {avg_loss:.6f} | "
                f"SDF: {avg_sdf_loss:.6f} | "
                f"Grad: {avg_grad_loss:.6f} | "
                f"Eikonal: {avg_eikonal_loss:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{args.mode}_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)

    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{args.mode}.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    total_time = time.time() - start_time
    print(f"Training finished. Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Final model saved to: {final_path}")


def train_all(args):
    """训练所有shape。"""
    uids = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    print(f"Found {len(uids)} shapes: {uids}")
    for uid in uids:
        print(f"\n{'='*60}")
        print(f"Training shape: {uid}")
        print(f"{'='*60}")
        args.uid = uid
        train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural SDF per shape")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--uid", type=str, default=None, help="指定训练单个shape的uid，不指定则训练所有")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "fourier"], help="训练模式")
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=10, help="隐藏层数量")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sine"], help="激活函数")
    parser.add_argument("--skip_layers", type=int, nargs="+", default=[5], help="skip connection层索引")
    parser.add_argument("--mapping_size", type=int, default=64, help="Fourier feature映射维度")
    parser.add_argument("--sigma", type=float, default=5.0, help="Fourier feature高斯矩阵标准差")
    parser.add_argument("--geometric_init", action="store_true", default=True, help="是否使用IGR几何初始化")
    parser.add_argument("--no_geometric_init", action="store_true", help="禁用几何初始化")
    parser.add_argument("--radius_init", type=float, default=1.0, help="几何初始化球面半径")
    parser.add_argument("--batch_size", type=int, default=8192, help="批次大小")
    parser.add_argument("--epochs", type=int, default=2000, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--lr_decay_step", type=int, default=500, help="学习率衰减步长")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="学习率衰减系数")
    parser.add_argument("--lambda_sdf", type=float, default=1.0, help="SDF损失权重")
    parser.add_argument("--lambda_grad", type=float, default=1.0, help="梯度损失权重")
    parser.add_argument("--lambda_eikonal", type=float, default=0.1, help="Eikonal损失权重")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument("--use_surface_points", action="store_true", help="是否加入表面点云进行训练")

    args = parser.parse_args()

    if args.no_geometric_init:
        args.geometric_init = False

    if args.uid is None:
        train_all(args)
    else:
        train(args)
