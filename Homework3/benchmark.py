"""Benchmark all models: forward/backward speed and FLOPs.

Usage:
    uv run python benchmark.py          # auto-detect device
    uv run python benchmark.py --device cuda
    uv run python benchmark.py --device mps
    uv run python benchmark.py --device cpu
"""
import argparse
import time

import torch
import torch.nn as nn

from models.pointnet2 import PointNet2ClsSSG, PointNet2ClsMSG
from models.pointnext import pointnext_s, pointnext_b


def count_flops(model, x):
    """Estimate FLOPs using torchprofile (if installed) or return -1."""
    try:
        from torchprofile import profile_macs
        return profile_macs(model, x)
    except ImportError:
        return -1


def benchmark(model, x, device, warmup=5, runs=20):
    """Benchmark forward and forward+backward latency."""
    # Warmup
    for _ in range(warmup):
        out = model(x)
        out.sum().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Forward only
    fwd_times = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Forward + backward
    fwb_times = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        t0 = time.perf_counter()
        model(x).sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        fwb_times.append((time.perf_counter() - t0) * 1000)

    fwd = sum(fwd_times) / len(fwd_times)
    fwb = sum(fwb_times) / len(fwb_times)
    return fwd, fwb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, help='cuda, mps, or cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-points', type=int, default=1024)
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Device: {device}')
    print(f'Batch size: {args.batch_size}, Num points: {args.num_points}')
    print()

    x = torch.randn(args.batch_size, args.num_points, 3, device=device)

    models = [
        ('PN++ SSG', lambda: PointNet2ClsSSG(num_classes=40)),
        ('PN++ MSG', lambda: PointNet2ClsMSG(num_classes=40)),
        ('PNXt-S',   lambda: pointnext_s(num_classes=40)),
        ('PNXt-B',   lambda: pointnext_b(num_classes=40)),
    ]

    print(f'{"Model":<12} {"Params":>8} {"FLOPs":>10} {"Fwd(ms)":>8} '
          f'{"F+B(ms)":>8} {"Samples/s":>10}')
    print('-' * 62)

    for name, build_fn in models:
        model = build_fn().to(device)
        params = sum(p.numel() for p in model.parameters())

        # FLOPs (on CPU with batch=1 for speed)
        flops = count_flops(model, torch.randn(1, args.num_points, 3))

        # Latency
        fwd, fwb = benchmark(model, x, device, runs=args.runs)
        throughput = args.batch_size / (fwb / 1000)

        flops_str = f'{flops/1e9:.2f}G' if flops > 0 else 'N/A'
        print(f'{name:<12} {params/1e6:>7.2f}M {flops_str:>10} {fwd:>7.1f}   '
              f'{fwb:>7.1f}   {throughput:>8.0f}')

        del model


if __name__ == '__main__':
    main()
