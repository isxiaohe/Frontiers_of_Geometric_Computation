# PointNet++ & PointNeXt Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce PointNet++ and PointNeXt on ModelNet40 classification with custom CUDA kernels + PyTorch fallbacks.

**Architecture:** Shared ops layer (`ops/`) with auto-selected CUDA/PyTorch backends. Two model directories (`models/pointnet2/`, `models/pointnext/`). Unified config-driven training script. CUDA kernel source files (`cuda_ops/csrc/`) are user-written; all C++ bindings and headers are provided.

**Tech Stack:** PyTorch 2.0+, CUDA 12+, einops, PyYAML, matplotlib

**Spec:** `docs/superpowers/specs/2026-04-27-pointnet2-pointnext-reproduction-design.md`

---

## Chunk 1: Project Scaffolding + Data Pipeline

### Task 1: Create directory structure and root files

**Files:**
- Create: `Homework3/ops/__init__.py`
- Create: `Homework3/ops/registry.py`
- Create: `Homework3/models/__init__.py`
- Create: `Homework3/models/pointnet2/__init__.py`
- Create: `Homework3/models/pointnext/__init__.py`
- Create: `Homework3/data/__init__.py`
- Create: `Homework3/configs/pointnet2_ssg.yaml`
- Create: `Homework3/configs/pointnet2_msg.yaml`
- Create: `Homework3/configs/pointnext.yaml`
- Create: `Homework3/requirements.txt`

- [ ] **Step 1: Create all directories**

```bash
cd Homework3
mkdir -p ops models/pointnet2 models/pointnext data configs cuda_ops/csrc cuda_ops/cuda_ops checkpoints docs
```

- [ ] **Step 2: Create `requirements.txt`**

```txt
torch>=2.0
numpy
pyyaml
matplotlib
h5py
tqdm
einops>=0.6.1
```

- [ ] **Step 3: Create `ops/registry.py`**

```python
try:
    import cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
```

- [ ] **Step 4: Create `ops/__init__.py`**

```python
from .registry import CUDA_AVAILABLE
from .ball_query import ball_query
from .fps import furthest_point_sample
from .group_points import group_points
from .interpolate import three_nn, three_interpolate, three_interpolation
```

- [ ] **Step 5: Create empty `models/__init__.py`, `models/pointnet2/__init__.py`, `models/pointnext/__init__.py`, `data/__init__.py`**

- [ ] **Step 6: Create `configs/pointnet2_ssg.yaml`**

```yaml
model: pointnet2_ssg
num_points: 1024
num_classes: 40
batch_size: 32
epochs: 200
lr: 0.001
weight_decay: 0.0001
optimizer: adam
scheduler: cosine
use_normals: false
augmentation:
  rotate_z: true
  jitter: true
  jitter_sigma: 0.01
  jitter_clip: 0.05
  dropout: true
  dropout_ratio: 0.875
```

- [ ] **Step 7: Create `configs/pointnet2_msg.yaml`**

```yaml
model: pointnet2_msg
num_points: 1024
num_classes: 40
batch_size: 16
epochs: 200
lr: 0.001
weight_decay: 0.0001
optimizer: adam
scheduler: cosine
use_normals: true
augmentation:
  rotate_z: true
  jitter: true
  jitter_sigma: 0.01
  jitter_clip: 0.05
  dropout: true
  dropout_ratio: 0.875
```

- [ ] **Step 8: Create `configs/pointnext.yaml`**

```yaml
model: pointnext_s
num_points: 1024
num_classes: 40
batch_size: 32
epochs: 300
lr: 0.001
weight_decay: 0.0001
optimizer: adamw
scheduler: cosine
use_normals: false
augmentation:
  rotate_z: true
  jitter: true
  jitter_sigma: 0.01
  jitter_clip: 0.05
  dropout: true
  dropout_ratio: 0.875
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding and config files"
```

---

### Task 2: ModelNet40 data pipeline

**Files:**
- Create: `Homework3/data/download.py`
- Create: `Homework3/data/modelnet40.py`

- [ ] **Step 1: Create `data/download.py`**

```python
"""Download and preprocess ModelNet40 dataset."""
import os
import urllib.request
import zipfile
from pathlib import Path


def download_modelnet40(root: str = "data/modelnet40"):
    """Download ModelNet40 from Princeton."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    url = "https://shapenet.cs.princeton.edu/media/modelnet40_ply_hdf5_2048.zip"
    zip_path = root / "modelnet40.zip"

    if not zip_path.exists():
        print(f"Downloading ModelNet40 to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists, skipping download.")

    if not (root / "ply_data_train.h5").exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(root)
        print("Extraction complete.")
    else:
        print("Data already extracted.")

    print(f"ModelNet40 ready at {root}")
```

- [ ] **Step 2: Create `data/modelnet40.py`**

```python
"""ModelNet40 PyTorch Dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path


class ModelNet40H5(Dataset):
    """ModelNet40 dataset from HDF5 files (pre-sampled 2048 points)."""

    def __init__(self, root, split='train', num_points=1024, normalize=True, augment=False):
        super().__init__()
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment

        h5_path = Path(root) / f"ply_data_{split}.h5"
        with h5py.File(h5_path, 'r') as f:
            self.data = f['data'][:]
            self.label = f['label'][:].flatten()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        pointcloud = self.data[idx].copy()
        label = self.label[idx]

        # Downsample
        if pointcloud.shape[0] > self.num_points:
            choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
            pointcloud = pointcloud[choice]

        xyz = pointcloud[:, :3]

        # Normalize to unit sphere
        if self.normalize:
            centroid = np.mean(xyz, axis=0)
            xyz = xyz - centroid
            max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
            if max_dist > 0:
                xyz = xyz / max_dist

        if self.augment:
            xyz = self._augment(xyz)

        return xyz.astype(np.float32), int(label)

    def _augment(self, xyz):
        # Random z-axis rotation
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        xyz = xyz @ R.T
        # Jitter
        xyz += np.clip(0.01 * np.random.randn(*xyz.shape), -0.05, 0.05)
        # Random point dropout
        keep = int(xyz.shape[0] * 0.875)
        idx = np.random.choice(xyz.shape[0], keep, replace=False)
        xyz = xyz[idx]
        return xyz


def build_dataloader(config):
    """Build train/test dataloaders from config."""
    root = config.get('data_root', 'data/modelnet40')
    num_points = config['num_points']
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)

    train_ds = ModelNet40H5(root, 'train', num_points, normalize=True, augment=True)
    test_ds = ModelNet40H5(root, 'test', num_points, normalize=True, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: ModelNet40 data pipeline with HDF5 loading and augmentation"
```

---

## Chunk 2: Ops Interface + PyTorch Fallbacks

### Task 3: Furthest Point Sampling op

**Files:**
- Create: `Homework3/ops/fps.py`

- [ ] **Step 1: Create `ops/fps.py`**

```python
"""Furthest Point Sampling - auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def furthest_point_sample(xyz, npoint):
    """
    Args:
        xyz: (B, N, 3) point coordinates
        npoint: int, number of points to sample
    Returns:
        indices: (B, npoint) int32
    """
    if CUDA_AVAILABLE:
        from cuda_ops.fps import furthest_point_sample_cuda
        return furthest_point_sample_cuda(xyz, npoint)
    return _fps_pytorch(xyz, npoint)


def _fps_pytorch(xyz, npoint):
    """Pure PyTorch FPS fallback."""
    device = xyz.device
    B, N, _ = xyz.shape
    indices = torch.zeros(B, npoint, dtype=torch.int32, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.int32, device=device)
    indices[:, 0] = farthest
    distances = torch.full((B, N), float('inf'), device=device)

    for i in range(1, npoint):
        centroid = xyz[torch.arange(B, device=device), farthest]  # (B, 3)
        dist = torch.sum((xyz - centroid.unsqueeze(1)) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances, dim=-1)
        indices[:, i] = farthest
    return indices
```

- [ ] **Step 2: Commit**

```bash
git add ops/fps.py
git commit -m "feat: FPS op with PyTorch fallback"
```

---

### Task 4: Ball Query op

**Files:**
- Create: `Homework3/ops/ball_query.py`

- [ ] **Step 1: Create `ops/ball_query.py`**

```python
"""Ball Query - auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Args:
        radius: float
        nsample: int
        xyz: (B, N, 3) source points
        new_xyz: (B, M, 3) query points
    Returns:
        idx: (B, M, nsample) int32
    """
    if CUDA_AVAILABLE:
        from cuda_ops.ball_query import ball_query_cuda
        return ball_query_cuda(radius, nsample, xyz, new_xyz)
    return _ball_query_pytorch(radius, nsample, xyz, new_xyz)


def _ball_query_pytorch(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz)  # (B, M, N)
    # Sort by distance, take first nsample within radius
    sorted_idx = dists.argsort(dim=-1)[:, :, :nsample]  # (B, M, nsample)
    sorted_dists = dists.gather(-1, sorted_idx)
    # Replace out-of-radius with first valid neighbor
    mask = sorted_dists > radius
    first_valid = sorted_idx[:, :, :1].expand_as(sorted_idx)
    idx = torch.where(mask, first_valid, sorted_idx)
    return idx.int()
```

- [ ] **Step 2: Commit**

```bash
git add ops/ball_query.py
git commit -m "feat: ball query op with PyTorch fallback"
```

---

### Task 5: Group Points op

**Files:**
- Create: `Homework3/ops/group_points.py`

- [ ] **Step 1: Create `ops/group_points.py`**

```python
"""Group Points (gather) - auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def group_points(features, idx):
    """
    Args:
        features: (B, C, N)
        idx: (B, M, K) int indices
    Returns:
        grouped: (B, C, M, K)
    """
    if CUDA_AVAILABLE:
        from cuda_ops.group_points import group_points_cuda
        return group_points_cuda(features, idx)
    return _group_points_pytorch(features, idx)


def _group_points_pytorch(features, idx):
    B, C, N = features.shape
    M, K = idx.shape[1], idx.shape[2]
    # Flatten idx, gather, reshape
    idx_flat = idx.unsqueeze(1).expand(B, C, M, K).reshape(B, C, M * K)
    grouped = features.gather(-1, idx_flat).reshape(B, C, M, K)
    return grouped
```

- [ ] **Step 2: Commit**

```bash
git add ops/group_points.py
git commit -m "feat: group points op with PyTorch fallback"
```

---

### Task 6: Three-NN Interpolation ops

**Files:**
- Create: `Homework3/ops/interpolate.py`

- [ ] **Step 1: Create `ops/interpolate.py`**

```python
"""Three-NN Interpolation - auto-selects CUDA or PyTorch."""
import torch
from .registry import CUDA_AVAILABLE


def three_nn(unknown, known):
    """
    Args:
        unknown: (B, N, 3) query points
        known: (B, M, 3) known points
    Returns:
        dist: (B, N, 3), idx: (B, N, 3) int32
    """
    if CUDA_AVAILABLE:
        from cuda_ops.interpolate import three_nn_cuda
        return three_nn_cuda(unknown, known)
    return _three_nn_pytorch(unknown, known)


def _three_nn_pytorch(unknown, known):
    dist2 = torch.cdist(unknown, known)  # (B, N, M)
    dist2, idx = dist2.topk(3, dim=-1, largest=False)
    return torch.sqrt(dist2), idx.int()


def three_interpolate(features, idx, weight):
    """
    Args:
        features: (B, C, M), idx: (B, N, 3), weight: (B, N, 3)
    Returns:
        out: (B, C, N)
    """
    if CUDA_AVAILABLE:
        from cuda_ops.interpolate import three_interpolate_cuda
        return three_interpolate_cuda(features, idx, weight)
    return _three_interpolate_pytorch(features, idx, weight)


def _three_interpolate_pytorch(features, idx, weight):
    B, C, M = features.shape
    N = idx.shape[1]
    idx_exp = idx.unsqueeze(1).expand(B, C, N, 3)  # (B, C, N, 3)
    gathered = features.gather(-1, idx_exp.reshape(B, C, N * 3)).reshape(B, C, N, 3)
    return (gathered * weight.unsqueeze(1)).sum(dim=-1)


def three_interpolation(known_xyz, known_feat, unknown_xyz):
    """Full pipeline: three_nn + weights + interpolate.
    Args:
        known_xyz: (B, 3, M), known_feat: (B, C, M), unknown_xyz: (B, 3, N)
    Returns: (B, C, N)
    """
    # Transpose to (B, N/M, 3) for three_nn
    known_t = known_xyz.transpose(1, 2).contiguous()
    unknown_t = unknown_xyz.transpose(1, 2).contiguous()
    dist, idx = three_nn(unknown_t, known_t)
    dist_recip = 1.0 / (dist + 1e-8)
    weight = dist_recip / dist_recip.sum(dim=-1, keepdim=True)
    return three_interpolate(known_feat, idx, weight)
```

- [ ] **Step 2: Commit**

```bash
git add ops/interpolate.py
git commit -m "feat: interpolation ops with PyTorch fallback"
```

---

### Task 7: Smoke test all ops on CPU

- [ ] **Step 1: Run smoke test**

```bash
cd Homework3
python -c "
import torch
from ops import furthest_point_sample, ball_query, group_points, three_interpolation

B, N, M = 2, 1024, 256
xyz = torch.randn(B, N, 3)
new_xyz = xyz[:, :M, :].clone()

idx = furthest_point_sample(xyz, 512)
print(f'FPS: {idx.shape}')  # (2, 512)

bi = ball_query(0.2, 32, xyz, new_xyz)
print(f'Ball query: {bi.shape}')  # (2, 256, 32)

feat = torch.randn(B, 64, N)
grouped = group_points(feat, bi)
print(f'Group: {grouped.shape}')  # (2, 64, 256, 32)

result = three_interpolation(torch.randn(B, 3, M), torch.randn(B, 64, M), torch.randn(B, 3, N))
print(f'Interp: {result.shape}')  # (2, 64, 1024)
print('All ops OK')
"
```

Expected: shapes printed, then "All ops OK"

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "test: verify all PyTorch fallback ops work on CPU"
```

---

## Chunk 3: CUDA Ops Scaffolding

### Task 8: CUDA ops headers

**Files:**
- Create: `Homework3/cuda_ops/csrc/cuda_utils.h`
- Create: `Homework3/cuda_ops/csrc/fps_gpu.h`
- Create: `Homework3/cuda_ops/csrc/ball_query_gpu.h`
- Create: `Homework3/cuda_ops/csrc/group_points_gpu.h`
- Create: `Homework3/cuda_ops/csrc/interpolate_gpu.h`

- [ ] **Step 1: Create `cuda_ops/csrc/cuda_utils.h`**

```c
#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H
#include <cmath>
#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}
#endif
```

- [ ] **Step 2: Create `cuda_ops/csrc/fps_gpu.h`**

```c
#ifndef _FPS_GPU_H
#define _FPS_GPU_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
void furthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs);
#endif
```

- [ ] **Step 3: Create `cuda_ops/csrc/ball_query_gpu.h`**

```c
#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H
#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
    const float *xyz, const float *new_xyz, int *idx);
#endif
```

- [ ] **Step 4: Create `cuda_ops/csrc/group_points_gpu.h`**

```c
#ifndef _GROUP_POINTS_GPU_H
#define _GROUP_POINTS_GPU_H
#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
void group_points_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample,
    const float *points, const int *idx, float *out);
void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample,
    const float *grad_out, const int *idx, float *grad_points);
#endif
```

- [ ] **Step 5: Create `cuda_ops/csrc/interpolate_gpu.h`**

```c
#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H
#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
    const float *known, float *dist2, int *idx);
void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n,
    const float *points, const int *idx, const float *weight, float *out);
void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m,
    const float *grad_out, const int *idx, const float *weight, float *grad_points);
#endif
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: CUDA op headers and utilities"
```

---

### Task 9: CUDA kernel stubs (USER WRITES THE KERNELS)

**Files:**
- Create: `Homework3/cuda_ops/csrc/fps_cuda.cu`
- Create: `Homework3/cuda_ops/csrc/ball_query_cuda.cu`
- Create: `Homework3/cuda_ops/csrc/group_points_cuda.cu`
- Create: `Homework3/cuda_ops/csrc/interpolate_cuda.cu`

These files contain the launcher boilerplate with TODO placeholders for the actual CUDA kernels. The user fills in the `__global__` kernel functions. See `docs/fps_guide.md` etc. for detailed instructions.

- [ ] **Step 1: Create `cuda_ops/csrc/fps_cuda.cu`** (stub with launcher, kernel marked TODO)

- [ ] **Step 2: Create `cuda_ops/csrc/ball_query_cuda.cu`** (stub)

- [ ] **Step 3: Create `cuda_ops/csrc/group_points_cuda.cu`** (stub)

- [ ] **Step 4: Create `cuda_ops/csrc/interpolate_cuda.cu`** (stub)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: CUDA kernel stubs with algorithm comments"
```

---

### Task 10: C++ dispatch, PyTorch bindings, setup.py

**Files:**
- Create: `Homework3/cuda_ops/csrc/fps.cpp`
- Create: `Homework3/cuda_ops/csrc/ball_query.cpp`
- Create: `Homework3/cuda_ops/csrc/group_points.cpp`
- Create: `Homework3/cuda_ops/csrc/interpolate.cpp`
- Create: `Homework3/cuda_ops/csrc/api.cpp`
- Create: `Homework3/cuda_ops/setup.py`
- Create: `Homework3/cuda_ops/__init__.py`
- Create: `Homework3/cuda_ops/fps.py`
- Create: `Homework3/cuda_ops/ball_query.py`
- Create: `Homework3/cuda_ops/group_points.py`
- Create: `Homework3/cuda_ops/interpolate.py`

- [ ] **Step 1: Create C++ dispatch files** (`fps.cpp`, `ball_query.cpp`, `group_points.cpp`, `interpolate.cpp`) — these unwrap ATen tensors and call the CUDA launchers.

- [ ] **Step 2: Create `api.cpp`** with `TORCH_LIBRARY(pointnet2_ops, ...)` registering all ops.

- [ ] **Step 3: Create `setup.py`** with `CUDAExtension` building all .cpp and .cu files.

- [ ] **Step 4: Create Python autograd wrappers** in `cuda_ops/` package — each op wraps `torch.autograd.Function` calling `_C.op_name(...)`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: CUDA ops C++ bindings, setup.py, and Python autograd wrappers"
```

---

## Chunk 4: Models

### Task 11: PointNet++ utility modules

**Files:**
- Create: `Homework3/models/pointnet2/pointnet_util.py`

- [ ] **Step 1: Create SA, SA_MSG, and FP modules** that use `ops/` for FPS, ball query, group points, and interpolation. Includes `PointNetSetAbstraction`, `PointNetSetAbstractionMsg`, `PointNetFeaturePropagation`.

- [ ] **Step 2: Commit**

---

### Task 12: PointNet++ classification models

**Files:**
- Create: `Homework3/models/pointnet2/pointnet2_cls.py`
- Modify: `Homework3/models/pointnet2/__init__.py`

- [ ] **Step 1: Create `PointNet2ClsSSG`** — 3 SA layers (512/128/1 points, radii 0.2/0.4/all, MLPs [64,64,128]/[128,128,256]/[256,512,1024]) + FC head (512->256->40).

- [ ] **Step 2: Create `PointNet2ClsMSG`** — 2 SA_MSG layers + 1 SA(global) + FC head.

- [ ] **Step 3: Update `__init__.py`** to export both classes.

- [ ] **Step 4: Smoke test** — verify model creation and forward pass with random input on CPU.

- [ ] **Step 5: Commit**

---

### Task 13: PointNeXt model

**Files:**
- Create: `Homework3/models/pointnext/pointnext_cls.py`
- Modify: `Homework3/models/pointnext/__init__.py`

- [ ] **Step 1: Create SABlock, InvResMLP, PointNextEncoder, PointNextClassification** — adapted from `submodules/pointnext/pointnext/pointnext.py` but using `ops/` instead of direct CUDA calls. Include `pointnext_s`, `pointnext_b`, `pointnext_l` factory functions.

- [ ] **Step 2: Update `__init__.py`** to export.

- [ ] **Step 3: Smoke test** — verify model creation and forward pass.

- [ ] **Step 4: Commit**

---

## Chunk 5: Training + Evaluation

### Task 14: Training script

**Files:**
- Create: `Homework3/train.py`

- [ ] **Step 1: Create `train.py`** — config-driven, builds model from YAML, Adam/AdamW optimizer, CosineAnnealing LR, CSV logging (epoch, train_loss, train_acc, test_loss, test_acc, lr), saves best checkpoint.

- [ ] **Step 2: Commit**

---

### Task 15: Evaluation script

**Files:**
- Create: `Homework3/test.py`

- [ ] **Step 1: Create `test.py`** — loads config + checkpoint, runs evaluation, prints per-class accuracy.

- [ ] **Step 2: Commit**

---

## Chunk 6: CUDA Kernel Guides + README

### Task 16: Algorithm guides for CUDA kernels

**Files:**
- Create: `Homework3/docs/fps_guide.md`
- Create: `Homework3/docs/ball_query_guide.md`
- Create: `Homework3/docs/group_points_guide.md`
- Create: `Homework3/docs/interpolate_guide.md`

Each guide covers:
- What the algorithm does
- Pseudocode
- CUDA thread mapping strategy
- Shared memory usage
- Reference to the implementation in `submodules/pointnext/csrc/`

- [ ] **Step 1: Write all 4 guides**

- [ ] **Step 2: Commit**

---

### Task 17: README

**Files:**
- Create: `Homework3/README.md`

- [ ] **Step 1: Write README** with: project overview, setup instructions, data download, training commands, evaluation commands, CUDA kernel building instructions.

- [ ] **Step 2: Final commit**
