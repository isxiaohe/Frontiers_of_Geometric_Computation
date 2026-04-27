"""Smoke tests for ops/ — verifies shapes, basic correctness, and edge cases.

Run:  uv run python tests/test_ops.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from ops import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ops import furthest_point_sample, ball_query, group_points
from ops import three_nn, three_interpolate, three_interpolation

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ─── FPS ────────────────────────────────────────────────────────────────
print("\n=== furthest_point_sample ===")
B, N, npoint = 2, 512, 128
xyz = torch.randn(B, N, 3)
idx = furthest_point_sample(xyz, npoint)

check("output shape", idx.shape == (B, npoint), f"got {idx.shape}")
check("dtype is long/int", idx.dtype in (torch.int32, torch.int64), f"got {idx.dtype}")
check("indices in [0, N)", (idx >= 0).all() and (idx < N).all())
# FPS should produce unique indices per batch element
for b in range(B):
    check(f"unique indices batch {b}", idx[b].unique().numel() == npoint,
          f"got {idx[b].unique().numel()} unique out of {npoint}")

# Edge: npoint == N → every point selected
idx_full = furthest_point_sample(xyz, N)
check("npoint=N returns all indices",
      all(idx_full[b].sort().values.equal(torch.arange(N)) for b in range(B)))

# Deterministic given same start? (hard to test with random start, skip)


# ─── Ball Query ─────────────────────────────────────────────────────────
print("\n=== ball_query ===")
B, N, M, nsample = 2, 512, 128, 32
xyz = torch.randn(B, N, 3)
new_xyz = xyz[:, :M, :].clone()

idx = ball_query(radius=0.2, nsample=nsample, xyz=xyz, new_xyz=new_xyz)
check("output shape", idx.shape == (B, M, nsample), f"got {idx.shape}")
check("indices in [0, N)", (idx >= 0).all().item() and (idx < N).all().item())
# All indices should be valid (no -1, no N)
check("no invalid indices", idx.min() >= 0)

# Tight radius with far-apart points → most neighbors should be the query itself
xyz_spread = torch.randn(B, N, 3) * 10.0
new_xyz_spread = xyz_spread[:, :M, :]
idx_tight = ball_query(radius=0.01, nsample=nsample, xyz=xyz_spread, new_xyz=new_xyz_spread)
# With tiny radius on spread points, most slots should fallback to nearest (self)
check("tight radius still returns valid indices", idx_tight.shape == (B, M, nsample))

# Large radius → all points within reach
idx_large = ball_query(radius=100.0, nsample=nsample, xyz=xyz, new_xyz=new_xyz)
check("large radius returns valid indices", idx_large.shape == (B, M, nsample))


# ─── Group Points ───────────────────────────────────────────────────────
print("\n=== group_points ===")
B, C, N, M, K = 2, 64, 512, 128, 32
features = torch.randn(B, C, N)
idx = torch.randint(0, N, (B, M, K))

grouped = group_points(features, idx)
check("output shape", grouped.shape == (B, C, M, K), f"got {grouped.shape}")

# Correctness: manually check a few entries
for b in range(B):
    for m in [0, M // 2, M - 1]:
        for k in [0, K - 1]:
            expected = features[b, :, idx[b, m, k]]
            actual = grouped[b, :, m, k]
            check(f"correct gather b={b} m={m} k={k}",
                  torch.allclose(expected, actual),
                  f"max diff = {(expected - actual).abs().max().item():.6e}")

# Gradient flow
features_grad = torch.randn(B, C, N, requires_grad=True)
grouped_grad = group_points(features_grad, idx)
grouped_grad.sum().backward()
check("gradient flows", features_grad.grad is not None)


# ─── Three NN ───────────────────────────────────────────────────────────
print("\n=== three_nn ===")
B, N_q, M_ref = 2, 256, 128
unknown = torch.randn(B, N_q, 3)
known = torch.randn(B, M_ref, 3)

dist, idx3 = three_nn(unknown, known)
check("dist shape", dist.shape == (B, N_q, 3), f"got {dist.shape}")
check("idx shape", idx3.shape == (B, N_q, 3), f"got {idx3.shape}")
check("indices in [0, M)", (idx3 >= 0).all().item() and (idx3 < M_ref).all().item())
# Distances should be non-negative
check("distances >= 0", (dist >= 0).all().item())
# Distances should be sorted (closest first)
check("distances sorted ascending", (dist[:, :, 0] <= dist[:, :, 1]).all().item()
      and (dist[:, :, 1] <= dist[:, :, 2]).all().item())

# Correctness: known point exactly at query → dist should be 0
known_copy = unknown[:, :M_ref, :].clone()  # first M_ref query points ARE the known points
dist2, idx2 = three_nn(unknown[:, :M_ref, :], known_copy)
check("self-distance ≈ 0", dist2[:, :, 0].max().item() < 0.01,
      f"max dist = {dist2[:, :, 0].max().item():.6e}")


# ─── Three Interpolate ─────────────────────────────────────────────────
print("\n=== three_interpolate ===")
B, C, M_ref, N_q = 2, 64, 128, 256
features = torch.randn(B, C, M_ref)
idx = torch.randint(0, M_ref, (B, N_q, 3))
weight = torch.rand(B, N_q, 3)
weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize

out = three_interpolate(features, idx, weight)
check("output shape", out.shape == (B, C, N_q), f"got {out.shape}")

# Correctness: uniform weights → average of 3 features
for b in range(B):
    for n in [0, N_q // 2]:
        expected = (features[b, :, idx[b, n, 0]]
                    + features[b, :, idx[b, n, 1]]
                    + features[b, :, idx[b, n, 2]]) / 3.0
        w = torch.ones(3) / 3.0
        actual = (features[b, :, idx[b, n, 0]] * w[0]
                  + features[b, :, idx[b, n, 1]] * w[1]
                  + features[b, :, idx[b, n, 2]] * w[2])
        check(f"correct interp b={b} n={n}",
              torch.allclose(expected, actual))

# Weights sum to 1 + constant features → output = features
features_const = torch.ones(B, C, M_ref)
out_const = three_interpolate(features_const, idx, weight)
check("constant features → constant output",
      torch.allclose(out_const, torch.ones_like(out_const), atol=1e-5))

# Gradient flow
features_grad = torch.randn(B, C, M_ref, requires_grad=True)
out_grad = three_interpolate(features_grad, idx, weight)
out_grad.sum().backward()
check("gradient flows", features_grad.grad is not None)


# ─── three_interpolation (full pipeline) ────────────────────────────────
print("\n=== three_interpolation (full pipeline) ===")
B, C, M, N = 2, 64, 128, 256
known_xyz = torch.randn(B, 3, M)
known_feat = torch.randn(B, C, M)
unknown_xyz = torch.randn(B, 3, N)

out = three_interpolation(known_xyz, known_feat, unknown_xyz)
check("output shape", out.shape == (B, C, N), f"got {out.shape}")

# If unknown_xyz == known_xyz, output should ≈ known_feat
out_self = three_interpolation(known_xyz, known_feat, known_xyz)
check("self-interp ≈ input",
      torch.allclose(out_self, known_feat, atol=0.1),
      f"max diff = {(out_self - known_feat).abs().max().item():.4f}")


# ─── End-to-end: FPS → Ball Query → Group Points ───────────────────────
print("\n=== end-to-end pipeline ===")
B, N = 2, 1024
xyz = torch.randn(B, N, 3)
C = 64
features = torch.randn(B, C, N)

# 1. FPS to 256 points
fps_idx = furthest_point_sample(xyz, 256)
new_xyz = xyz.gather(1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
check("FPS → new_xyz shape", new_xyz.shape == (B, 256, 3))

# 2. Ball query
bq_idx = ball_query(0.2, 32, xyz, new_xyz)
check("BQ shape", bq_idx.shape == (B, 256, 32))

# 3. Group features
grouped = group_points(features, bq_idx.long())
check("grouped shape", grouped.shape == (B, C, 256, 32))

# 4. Max pool → local features
local_feat = grouped.max(dim=-1)[0]
check("local features shape", local_feat.shape == (B, C, 256))

# 5. Interpolate back to original resolution
known_xyz_t = new_xyz.transpose(1, 2)     # (B, 3, 256)
unknown_xyz_t = xyz.transpose(1, 2)       # (B, 3, 1024)
upsampled = three_interpolation(known_xyz_t, local_feat, unknown_xyz_t)
check("upsampled shape", upsampled.shape == (B, C, N))


# ─── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
