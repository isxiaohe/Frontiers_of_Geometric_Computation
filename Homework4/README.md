# Homework4 - Neural SDF Surface Reconstruction

基于MLP的点云表面重建（Neural SDF）。

## 环境要求

- Python 3.10+
- PyTorch
- numpy
- scikit-image
- trimesh

## 安装

```bash
uv pip install torch numpy scikit-image trimesh
```

## 文件结构

```
model.py        # SDFMLP 基础模型 + FourierFeatureMLP 拓展模型
dataset.py      # 加载单个 shape 的 SDF 采样数据为 PyTorch Dataset
train.py        # 逐 shape 训练脚本，支持基础/拓展模式切换
test.py         # 逐 shape 测试脚本，网格采样 + Marching Cubes
utils.py        # mesh 导出、Chamfer Distance、IoU 计算等工具函数
README.md       # 运行说明
```

## 训练

### 基础模型

训练单个shape：
```bash
python train.py --mode base --uid 1a04e3eab45ca15dd86060f189eb133
```

训练所有shape：
```bash
python train.py --mode base
```

### Fourier Feature 拓展模型

```bash
python train.py --mode fourier --uid 1a04e3eab45ca15dd86060f189eb133
```

### 主要超参数

- `--hidden_dim`: 隐藏层维度（默认256）
- `--num_layers`: 隐藏层数量（默认8）
- `--activation`: 激活函数 relu/sine（默认relu）
- `--batch_size`: 批次大小（默认8192）
- `--epochs`: 训练轮数（默认2000）
- `--lr`: 学习率（默认1e-3）
- `--lr_decay_step`: 学习率衰减步长（默认500）
- `--lambda_grad`: 梯度损失权重（默认0.1）
- `--mapping_size`: Fourier feature映射维度（默认10）
- `--sigma`: Fourier feature高斯矩阵标准差（默认10.0）

## 测试

### 提取Mesh

```bash
python test.py --mode base --uid 1a04e3eab45ca15dd86060f189eb133
```

输出保存在 `results/<uid>_<mode>.obj`。

### 主要参数

- `--resolution`: Marching Cubes网格分辨率（默认128）
- `--batch_size`: 推理批次大小（默认65536）

## 训练策略

- 每个shape单独训练一个网络
- 损失函数：`L = MSE(sdf_pred, sdf_gt) + λ * MSE(grad_pred, grad_gt)`
- 优化器：Adam，lr=1e-3
- 学习率衰减：每500 epoch衰减0.5
- Batch size：8192
- Epochs：2000

## 结果

训练和测试的日志、模型检查点分别保存在：
- `checkpoints/`: 模型参数文件 `.pt`
- `results/`: 重建的mesh文件 `.obj`
