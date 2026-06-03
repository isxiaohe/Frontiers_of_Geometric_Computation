import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SDFMLP, GaussianFourierFeatureTransform
from dataset import RecDataset


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
    dataset = RecDataset(
        data_dir,
        mode=args.mode,
        sample_size=args.sample_size,
        mix_ratio=args.mix_ratio,
        num_iters=args.num_iters,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if args.use_fourier:
        fourier_transform = GaussianFourierFeatureTransform(
            in_dim=3, mapping_size=args.mapping_size, sigma=args.sigma
        )
        model = SDFMLP(
            in_dim=2 * args.mapping_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=args.activation,
            skip_layers=args.skip_layers,
            fourier_transform=fourier_transform,
            geometric_init=args.geometric_init,
            radius_init=args.radius_init,
        ).to(device)
    else:
        model = SDFMLP(
            in_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=args.activation,
            skip_layers=args.skip_layers,
            geometric_init=args.geometric_init,
            radius_init=args.radius_init,
        ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=1e-6
    )

    model_tag = f"{args.mode}{'_fourier' if args.use_fourier else ''}"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{model_tag}_log.json")
    log = []

    # Optional tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=os.path.join(args.checkpoint_dir, f"{args.uid}_{model_tag}_tb")
        )
    except ImportError:
        writer = None

    model.train()
    start_time = time.time()

    pbar = tqdm(enumerate(dataloader, 1), total=args.num_iters, desc="Training", ncols=100)
    for it, batch in pbar:
        # DataLoader adds batch dim -> squeeze it
        batch = {k: v.squeeze(0) for k, v in batch.items()}
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
            empty_points = (
                torch.rand(points.shape[0] // 2, 3, device=device, dtype=points.dtype)
                - 0.5
            )
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
        scheduler.step()

        log.append({
            "iter": it,
            "loss": loss.item(),
            "sdf_loss": loss_sdf.item(),
            "grad_loss": loss_grad.item(),
            "eikonal_loss": loss_eikonal.item(),
            "lr": scheduler.get_last_lr()[0],
        })

        postfix = {
            "loss": f"{loss.item():.6f}",
            "sdf": f"{loss_sdf.item():.6f}",
            "grad": f"{loss_grad.item():.6f}",
        }
        if args.lambda_eikonal > 0:
            postfix["eikonal"] = f"{loss_eikonal.item():.6f}"
        postfix["lr"] = f"{scheduler.get_last_lr()[0]:.6f}"
        pbar.set_postfix(postfix)

        if writer is not None:
            writer.add_scalar("Loss/total", loss.item(), it)
            writer.add_scalar("Loss/sdf", loss_sdf.item(), it)
            writer.add_scalar("Loss/grad", loss_grad.item(), it)
            if args.lambda_eikonal > 0:
                writer.add_scalar("Loss/eikonal", loss_eikonal.item(), it)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], it)

        if it % args.log_interval == 0 or it == 1:
            elapsed = time.time() - start_time
            print(
                f"Iter {it}/{args.num_iters} | "
                f"Loss: {loss.item():.6f} | "
                f"SDF: {loss_sdf.item():.6f} | "
                f"Grad: {loss_grad.item():.6f} | "
                f"Eikonal: {loss_eikonal.item():.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        if it % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.uid}_{model_tag}_iter{it}.pt"
            )
            torch.save(
                {
                    "iter": it,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )

    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{model_tag}.pt")
    torch.save(
        {
            "iter": args.num_iters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    print(f"Training finished. Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Final model saved to: {final_path}")


def train_all(args):
    """训练所有shape。"""
    uids = sorted(
        [
            d
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ]
    )
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
    parser.add_argument(
        "--uid", type=str, default=None, help="指定训练单个shape的uid，不指定则训练所有"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["sdf", "surface", "mixed"],
        help="数据采样模式",
    )
    parser.add_argument(
        "--use_fourier", action="store_true", help="是否使用Fourier feature位置编码"
    )
    parser.add_argument("--hidden_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=10, help="隐藏层数量")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sine"],
        help="激活函数",
    )
    parser.add_argument(
        "--skip_layers", type=int, nargs="+", default=[5], help="skip connection层索引"
    )
    parser.add_argument(
        "--mapping_size", type=int, default=64, help="Fourier feature映射维度"
    )
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="Fourier feature高斯矩阵标准差"
    )
    parser.add_argument(
        "--geometric_init", action="store_true", default=True, help="是否使用IGR几何初始化"
    )
    parser.add_argument(
        "--no_geometric_init", action="store_true", help="禁用几何初始化"
    )
    parser.add_argument(
        "--radius_init", type=float, default=1.0, help="几何初始化球面半径"
    )
    parser.add_argument(
        "--num_iters", type=int, default=100000, help="训练迭代次数"
    )
    parser.add_argument(
        "--sample_size", type=int, default=25000, help="每iteration采样点数"
    )
    parser.add_argument(
        "--mix_ratio", type=float, default=0.5, help="mixed模式下sdf点比例"
    )
    parser.add_argument("--lr", type=float, default=None, help="学习率 (base默认5e-4, fourier默认2e-4)")
    parser.add_argument("--lambda_sdf", type=float, default=None, help="SDF损失权重 (base默认1.0, fourier默认2.0)")
    parser.add_argument("--lambda_grad", type=float, default=None, help="梯度损失权重 (base默认1.0, fourier默认0.5)")
    parser.add_argument(
        "--lambda_eikonal", type=float, default=None, help="Eikonal损失权重 (base默认0.1, fourier默认0.0)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="模型保存目录"
    )
    parser.add_argument("--log_interval", type=int, default=1000, help="日志打印间隔")
    parser.add_argument(
        "--save_interval", type=int, default=20000, help="模型保存间隔"
    )

    args = parser.parse_args()

    if args.no_geometric_init:
        args.geometric_init = False

    # 自动设置未显式指定的参数
    if args.use_fourier:
        if args.lr is None:
            args.lr = 2e-4
        if args.lambda_sdf is None:
            args.lambda_sdf = 2.0
        if args.lambda_grad is None:
            args.lambda_grad = 0.5
        if args.lambda_eikonal is None:
            args.lambda_eikonal = 0.0
    else:
        if args.lr is None:
            args.lr = 5e-4
        if args.lambda_sdf is None:
            args.lambda_sdf = 1.0
        if args.lambda_grad is None:
            args.lambda_grad = 1.0
        if args.lambda_eikonal is None:
            args.lambda_eikonal = 0.1

    print(f"Training config: lr={args.lr}, lambda_sdf={args.lambda_sdf}, lambda_grad={args.lambda_grad}, lambda_eikonal={args.lambda_eikonal}, geometric_init={args.geometric_init}")

    if args.uid is None:
        train_all(args)
    else:
        train(args)
