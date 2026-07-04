# Tiny 3D Prediction Target Baseline

This folder now contains a minimal runnable point-cloud generation baseline for
testing `x0`, `epsilon`, and `v` prediction targets under flow matching.

The implementation is intentionally small and independent from `submodules/`.
It uses synthetic chair-like point clouds so the smoke tests and CLI do not need
ShapeNet downloads.

This is a baseline/smoke-test codebase, not a full 3D generation benchmark.
Generation summaries report lightweight Chamfer metrics (`mmd_cd`,
`coverage`) and visualization grids; report text should avoid comparing the
absolute generation quality directly against full-scale 3D diffusion methods.
For target comparisons, `run_primitives_generation_experiment` fixes the model
initialization, dataloader order, training randomness, and initial sampling
noise seed across `x0`, `epsilon`, and `v`; only the prediction target changes.

## Files

- `src/fgc3d/targets.py`: VP/DDPM alpha/sigma schedule plus `x0`, `epsilon`,
  and `v` target conversions for diffusion reference runs.
- `src/fgc3d/flow_matching.py`: OT flow path, FM-specific target conversions,
  and Euler ODE sampling.
- `src/fgc3d/model.py`: pointwise MLP, PointNet-context, and a pure-PyTorch
  PVConv/PVCNN-style voxel+point denoisers with sinusoidal time embeddings.
- `src/fgc3d/device.py`: explicit `auto`/`cpu`/`cuda[:i]` device resolution for
  local smoke tests and server training.
- `src/fgc3d/metrics.py`: lightweight Chamfer, MMD-CD, and Coverage-CD metrics
  for small generated/reference point-cloud sets.
- `src/fgc3d/data.py`: deterministic synthetic point-cloud datasets plus a
  ShapeNetCore.v2.PC15k-style single-category loader.
- `src/fgc3d/train.py`: train step and short toy training loop.
- `src/fgc3d/scheduler.py`: cosine-VP DDIM scheduler and PVD-style discrete
  DDPM beta scheduler for reference/sanity checks.
- `src/fgc3d/sample.py`: deterministic DDIM-style sampling smoke path.
- `src/fgc3d/overfit.py`: fixed-batch overfitting diagnostic.
- `src/fgc3d/cli/train_toy.py`: CLI entrypoint for short training runs.
- `tests/`: standard-library `unittest` coverage for targets, model, training,
  CLI, and sampling.

## Server Quickstart

On an NVIDIA server, recreate the environment instead of copying `.venv`:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-base.txt
uv pip install -r env/requirements-cuda-cu128.txt
```

Replace `cu128` with the PyTorch wheel index matching the server CUDA runtime if
needed. Then verify that the project and CUDA are visible:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_env.py --output env/current-env.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The training scripts default to `--device auto`, which uses CUDA when
`torch.cuda.is_available()` is true and otherwise falls back to CPU. You can pin
a GPU explicitly with `--device cuda:0`.

For a first server run on the downloaded chair subset:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_shapenet_category.py \
  --output-dir outputs/server_chair_pvdlite_flow_x0 \
  --data-root data/ShapeNetCore.v2.PC15k-mini \
  --category chair \
  --objective flow \
  --model pvdlite \
  --prediction-target x0 \
  --loss-mode target \
  --steps 2000 \
  --batch-size 16 \
  --num-points 1024 \
  --hidden-dim 64 \
  --voxel-resolution 8 \
  --num-samples 8 \
  --sample-steps 64 \
  --device auto
```

For the core target-comparison run on synthetic finite sphere/ellipsoid data:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_primitives.py \
  --output-dir outputs/server_primitives_pvdlite_flow_xev \
  --objective flow \
  --targets x0 epsilon v \
  --model pvdlite \
  --loss-mode target \
  --steps 2000 \
  --batch-size 64 \
  --num-shapes 1024 \
  --num-points 256 \
  --hidden-dim 64 \
  --voxel-resolution 8 \
  --num-samples 16 \
  --sample-steps 64 \
  --device auto
```

Each run writes `summary.json`, generated sample tensors, model weights, and a
visualization grid under the selected `outputs/...` directory.

## UV Environment

Detailed migration notes live in `env/README.md`.

Create an independent local environment:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-cpu.txt
```

For an NVIDIA CUDA machine, install the PyTorch wheel from the CUDA index that
matches the target machine. For example, for CUDA 12.8 wheels:

```bash
uv venv --python 3.12 .venv
uv pip install -r env/requirements-base.txt
uv pip install -r env/requirements-cuda-cu128.txt
```

Do not use `--system-site-packages` for a portable training environment. It was
only used once on this Mac as a no-download verification fallback when sandbox
policy blocked installing PyTorch into `.venv`.

Record the active environment after setup:

```bash
PYTHONPATH=src .venv/bin/python scripts/record_env.py --output env/current-env.json
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Run a short training smoke test:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --prediction-target x0 \
  --steps 5 \
  --batch-size 4
```

Replace `x0` with `epsilon` or `v` to test the other prediction targets.

`--loss-mode target` compares the model output to its own prediction target.
`--loss-mode x0`, `--loss-mode epsilon`, and `--loss-mode v` first convert the
model output into that common target space, then compute MSE there.

For the OT flow path, `epsilon` means the source noise endpoint `z`, not the
same object as DDPM epsilon under a VP schedule. During sampling, converting an
endpoint prediction back to velocity can amplify errors near `t=0` for
`epsilon` and near `t=1` for `x0`. Generation summaries therefore include
`sampling_condition_number`; `v` prediction has condition number `1` on this
path.

Run the normalized finite sphere/ellipsoid point-cloud toy:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective flow \
  --dataset primitives \
  --prediction-target x0 \
  --steps 200
```

Run a no-download structural toy with synthetic `chair`, `airplane`, and
`table` point clouds:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective flow \
  --dataset structural \
  --model pvdlite \
  --voxel-resolution 6 \
  --prediction-target x0 \
  --steps 200
```

Run the same toy with the PVD/PVConv-inspired lightweight voxel+point model:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective flow \
  --dataset primitives \
  --model voxelpoint \
  --voxel-resolution 6 \
  --prediction-target epsilon \
  --steps 120
```

Run the same model with a PVD-style discrete DDPM schedule as a diffusion
reference baseline:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective diffusion \
  --dataset primitives \
  --model voxelpoint \
  --voxel-resolution 6 \
  --prediction-target epsilon \
  --diffusion-schedule ddpm \
  --ddpm-steps 1000 \
  --steps 120
```

Run the stronger PVD-lite model with global context:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective flow \
  --dataset primitives \
  --model pvdlite \
  --voxel-resolution 6 \
  --prediction-target x0 \
  --steps 200
```

Run a fixed-batch overfit diagnostic for the voxel+point model:

```bash
PYTHONPATH=src .venv/bin/python scripts/overfit_primitives.py \
  --objective flow \
  --model voxelpoint \
  --prediction-target x0 \
  --loss-mode target \
  --steps 200 \
  --batch-size 2 \
  --num-points 64
```

Generate side-by-side `x0`, `epsilon`, and `v` primitive samples:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_primitives.py \
  --output-dir outputs/primitives_pvdlite_flow_xev_120 \
  --objective flow \
  --model pvdlite \
  --voxel-resolution 6 \
  --steps 120 \
  --hidden-dim 32 \
  --device auto
```

For a fast DDPM generation smoke test, `--ddpm-steps` sets the training beta
schedule length and `--sample-steps` controls the number of reverse sampling
updates:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_primitives.py \
  --output-dir outputs/primitives_voxelpoint_ddpm_smoke \
  --objective diffusion \
  --model voxelpoint \
  --diffusion-schedule ddpm \
  --ddpm-steps 16 \
  --sample-steps 8 \
  --steps 20 \
  --hidden-dim 24 \
  --num-points 24 \
  --no-plots
```

For a longer primitive x0-generation check on the FM main path:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_primitives.py \
  --output-dir outputs/primitives_pvdlite_flow_x0_800 \
  --objective flow \
  --targets x0 \
  --model pvdlite \
  --voxel-resolution 6 \
  --steps 800 \
  --batch-size 16 \
  --num-shapes 256 \
  --num-points 64 \
  --hidden-dim 32
```

Run the same sphere/ellipsoid toy with OT flow matching:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --objective flow \
  --dataset primitives \
  --prediction-target v \
  --steps 500
```

Run the Back-to-Basics-style projected unit-sphere toy with unified `v` loss:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_manifold \
  --prediction-target x0 \
  --loss-mode v \
  --intrinsic-dim 2 \
  --ambient-dim 512
```

Prepare one ShapeNetCore category from Hugging Face. The dataset is gated, so
first run `huggingface-cli login` and make sure access to
`ShapeNet/ShapeNetCore` has been approved. This command downloads one category
archive and converts only the first 16 chair meshes into point-cloud `.npy`
files. `airplane` and `table` are also good visual categories for this project;
`mug` is less useful for early debugging because the handle and interior are
hard to judge in sparse point-cloud plots.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_shapenetcore_category.py \
  --download \
  --category chair \
  --max-models 16 \
  --num-points 2048 \
  --output-root data/ShapeNetCore.v2.PC15k-mini \
  --cache-dir .cache/huggingface
```

Run one ShapeNet category when a PC15k-style dataset is available locally:

```bash
PYTHONPATH=src .venv/bin/python -m fgc3d.cli.train_toy \
  --dataset shapenet \
  --data-root data/ShapeNetCore.v2.PC15k-mini \
  --category chair \
  --objective flow \
  --model pvdlite \
  --voxel-resolution 6 \
  --prediction-target x0 \
  --steps 200 \
  --batch-size 2 \
  --num-points 256
```

Train and save generated ShapeNet category samples:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_generate_shapenet_category.py \
  --output-dir outputs/shapenet_chair_pvdlite_flow_x0_200 \
  --data-root data/ShapeNetCore.v2.PC15k-mini \
  --category chair \
  --objective flow \
  --model pvdlite \
  --prediction-target x0 \
  --steps 200 \
  --batch-size 2 \
  --num-points 256 \
  --hidden-dim 32
```

Expected dataset layout:

```text
data/ShapeNetCore.v2.PC15k-mini/
  03001627/
    train/*.npy
    val/*.npy
    test/*.npy
```

The category can be a ShapeNet name such as `chair`, `airplane`, `table`, or
the synset id itself such as `03001627`.
