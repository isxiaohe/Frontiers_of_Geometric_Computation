# Neural SDF Surface Reconstruction

基于 MLP 的点云隐式表面重建（Neural SDF），支持 Base MLP 与 Fourier Feature MLP 两种模式。

## 环境

```bash
# Python 3.10+
uv pip install torch numpy scikit-image trimesh tqdm
```

## 训练

### Base MLP

```bash
python train.py \
  --uid <SHAPE_ID> \
  --mode surface \
  --num_iters 10000 \
  --sample_size 25000 \
  --lr 5e-4 \
  --lambda_sdf 1.0 \
  --lambda_grad 1.0 \
  --lambda_eikonal 0.1 \
  --checkpoint_dir checkpoints/base
```

### Fourier Feature MLP

```bash
python train.py \
  --uid <SHAPE_ID> \
  --mode surface \
  --use_fourier \
  --num_iters 20000 \
  --sample_size 25000 \
  --mapping_size 64 \
  --sigma 5.0 \
  --lr 2e-4 \
  --lambda_sdf 2.0 \
  --lambda_grad 0.5 \
  --lambda_eikonal 0.0 \
  --checkpoint_dir checkpoints/fourier
```

> 省略 `--uid` 即可训练 `data/` 目录下的全部 shape。

## 测试

### Base MLP

```bash
python test.py \
  --uid <SHAPE_ID> \
  --mode surface \
  --resolution 256 \
  --checkpoint_dir checkpoints/base \
  --result_dir results/base
```

### Fourier Feature MLP

```bash
python test.py \
  --uid <SHAPE_ID> \
  --mode surface \
  --use_fourier \
  --resolution 256 \
  --clean_mesh \
  --checkpoint_dir checkpoints/fourier \
  --level 0.0001 \
  --result_dir results/fourier
```

> 省略 `--uid` 即可测试全部 shape。`--clean_mesh` 会保留最大连通组件，去除漂浮面片。
