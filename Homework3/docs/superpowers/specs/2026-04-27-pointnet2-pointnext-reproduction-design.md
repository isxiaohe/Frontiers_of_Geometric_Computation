# PointNet++ & PointNeXt ModelNet40 Classification Reproduction

## Goal

Reproduce two 3D point cloud classification methods on ModelNet40 for the Frontiers of Geometric Computation homework:
1. **PointNet++** (SSG and MSG variants)
2. **PointNeXt** (using InvResMLP blocks)

The project uses custom CUDA kernels (written by the user as a learning exercise) with PyTorch fallbacks for development on Mac. Training runs on an NVIDIA RTX 4060 laptop.

## Architecture

### Project Structure

```
Homework3/
├── cuda_ops/                    # Custom CUDA kernels (user-written)
│   ├── setup.py                 # torch.utils.cpp_extension build
│   ├── csrc/
│   │   ├── api.cpp              # PyTorch extension bindings (TORCH_LIBRARY)
│   │   ├── cuda_utils.h         # CUDA_CHECK macros, common utilities
│   │   ├── ball_query.cpp       # C++/CUDA dispatch
│   │   ├── ball_query_cuda.cu
│   │   ├── fps.cpp
│   │   ├── fps_cuda.cu
│   │   ├── group_points.cpp
│   │   ├── group_points_cuda.cu
│   │   ├── interpolate.cpp
│   │   └── interpolate_cuda.cu
│   └── cuda_ops/                # Python package with autograd wrappers
│       ├── __init__.py
│       ├── ball_query.py
│       ├── fps.py
│       ├── group_points.py
│       └── interpolate.py
│
├── ops/                         # Unified op interface
│   ├── __init__.py              # Auto-selects CUDA or PyTorch
│   ├── registry.py              # try_cuda() / fallback detection
│   ├── ball_query.py
│   ├── fps.py
│   ├── group_points.py
│   └── interpolate.py
│
├── models/
│   ├── __init__.py
│   ├── pointnet2/
│   │   ├── __init__.py
│   │   ├── pointnet2_cls.py     # SSG + MSG classification models
│   │   └── pointnet_util.py     # SA module, FP module
│   └── pointnext/
│       ├── __init__.py
│       └── pointnext.py         # PointNeXt classification model
│
├── data/
│   ├── __init__.py
│   ├── modelnet40.py            # PyTorch Dataset
│   └── download.py              # Download + preprocess
│
├── train.py                     # Unified training script
├── test.py                      # Evaluation script
├── configs/
│   ├── pointnet2_ssg.yaml
│   ├── pointnet2_msg.yaml
│   └── pointnext.yaml
├── docs/                        # Algorithm guides for CUDA kernel writing
│   ├── fps_guide.md
│   ├── ball_query_guide.md
│   ├── group_points_guide.md
│   └── interpolate_guide.md
├── README.md
└── report/
```

### Op Selection Pattern

Every geometric op has two implementations:
1. **CUDA** (in `cuda_ops/`) — fast, runs on NVIDIA GPU
2. **PyTorch** (in `ops/`) — fallback for Mac development, pure tensor ops

Selection is automatic at import time:
```python
# ops/registry.py
try:
    import cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
```

Each op module exposes a single function that dispatches accordingly.

## CUDA Kernels

Four kernels shared by both models. Each is wrapped in `torch.autograd.Function` with forward and backward passes.

### 1. Farthest Point Sampling (FPS)

- **Input**: point cloud (B, N, 3), number of output points M
- **Output**: indices (B, M)
- **Algorithm**: Iteratively select the point farthest from all already-selected points. Sequential in nature — each step depends on the previous.
- **CUDA strategy**: Parallelize across batch dimension. Each block handles one sample. Maintain a distance array in shared memory, update after each selection.
- **Difficulty**: Hard (sequential dependency)
- **Gradient**: None (non-differentiable, used as index operation)

### 2. Ball Query

- **Input**: radius r, max neighbors K, source points (B,N,3), query points (B,M,3)
- **Output**: neighbor indices (B, M, K)
- **Algorithm**: For each query point, find all points within radius r. If fewer than K found, pad with the first found point.
- **CUDA strategy**: One thread per query point. Iterate over source points, compute squared distance, collect indices within radius.
- **Difficulty**: Medium
- **Gradient**: None (non-differentiable index operation)

### 3. Group Points (Gather)

- **Input**: features (B, C, N), indices (B, M, K)
- **Output**: grouped features (B, C, M, K)
- **Algorithm**: Gather features at the given indices.
- **CUDA strategy**: One thread per (batch, channel, query_point, neighbor) tuple. Direct memory read.
- **Difficulty**: Easy
- **Gradient**: Backward scatters gradients to the original feature positions (atomic add or indexed assignment).

### 4. Three-NN Interpolation

- **Input**: known features (B, C, M), known positions (B, M, 3), unknown positions (B, N, 3)
- **Output**: interpolated features (B, C, N)
- **Algorithm**: For each unknown point, find 3 nearest known points, compute inverse-distance weights, weighted sum.
- **CUDA strategy**: One thread per unknown point. Compute distances to all known points, maintain top-3.
- **Difficulty**: Medium
- **Gradient**: Backward propagates gradients to known features weighted by the interpolation weights.

## Model Architectures

### PointNet++ Classification

**Set Abstraction (SA) Layer**:
1. FPS to select M centroids from N points
2. Ball query to find K neighbors per centroid
3. Group neighbor features
4. MLP on grouped features (shared across neighbors)
5. Max pooling across neighbors → (B, C', M)

**SSG (Single-Scale Grouping)**:
```
Input (B, N, 3)
  → SA(512, 0.2, 32, [64,64,128])    → (B, 512, 128)
  → SA(128, 0.4, 64, [128,128,256])   → (B, 128, 256)
  → SA(1,   None, None, [256,512,1024]) → (B, 1, 1024)  [global]
  → FC(512, 256, 40)
```

**MSG (Multi-Scale Grouping)**:
Same structure but each SA layer runs ball query at multiple radii and concatenates features.

### PointNeXt Classification

**InvResMLP Block**:
- Inverted residual design (like MobileNetV2)
- Depthwise separable convolution on point neighborhoods
- Skip connections

**Architecture**:
```
Input (B, N, 3)
  → Stem MLP
  → [InvResMLP blocks at multiple resolutions via SA layers]
  → Global average pooling
  → FC head → 40 classes
```

PointNeXt reuses the same SA layer pattern (FPS + ball query + group + MLP) but with the InvResMLP block replacing the plain MLP.

## Training Pipeline

### Dataset: ModelNet40
- 9,843 training samples, 2,468 test samples
- 40 categories
- 1,024 points per sample (uniformly sampled from mesh surface)
- Standard splits from the original dataset

### Data Augmentation
- Random rotation around z-axis
- Random jitter (Gaussian noise on coordinates)
- Random point dropout
- Normalization to unit sphere

### Training Configuration
- **Optimizer**: Adam, lr=0.001
- **LR schedule**: Cosine annealing or step decay
- **Batch size**: 32 (fits in RTX 4060 8GB)
- **Epochs**: 200
- **Loss**: Cross-entropy

### Logging
- Per-epoch train/test loss and accuracy saved to CSV
- Auto-generate loss/accuracy curves as PNG plots
- Save best model checkpoint

### Config-driven
Each experiment has a YAML config specifying:
```yaml
model: pointnet2_ssg
num_points: 1024
batch_size: 32
epochs: 200
lr: 0.001
optimizer: adam
augmentation: [rotate_z, jitter, dropout]
```

## Homework Deliverables Mapping

| Deliverable | Where |
|---|---|
| Code implementation (3 pts x2) | cuda_ops/ + models/ + train.py + test.py |
| Method introduction (1 pt x2) | In report, based on docs/ guides |
| Loss/Accuracy curves + comparison (2 pts x2) | Auto-generated from training logs |
| Comparative analysis (3 pts) | In report: parameter count, speed, accuracy table |
| README | Root README.md with run instructions |
| Trained model weights | checkpoints/ directory |

## Development Workflow

1. **On Mac**: Develop models, data pipeline, training script. PyTorch fallback ops work on CPU/MPS.
2. **On RTX 4060**: Write CUDA kernels, train models. Build `cuda_ops` with `pip install .`
3. **Docs**: Algorithm guides in `docs/` help user understand and implement CUDA kernels.

## Dependencies

```
torch >= 2.0
numpy
pyyaml
matplotlib
h5py
tqdm
einops
```
