# 基于神经网络的三维表面重建

## 1. Base MLP 表面重建

### 1.1 模型结构与方法

#### 网络架构

我们采用基于 IGR (Implicit Geometric Regularization) 的多层感知机 (MLP) 作为基础模型，记为 SDFMLP。网络结构如下：

- 输入：三维空间坐标 $q \in \mathbb{R}^3$
- 输出：标量 SDF 值 $\tilde{F} \in \mathbb{R}$
- 隐藏层维度：$\text{hidden\_dim} = 512$
- 隐藏层数量：$10$ 层（含输出层共 $11$ 层）
- 激活函数：ReLU
- Skip Connection：第 $5$ 层引入 skip connection，将原始输入 $q$ 与隐藏特征拼接后除以 $\sqrt{2}$ 以保持方差稳定

#### 几何初始化

参考 IGR 论文，我们采用几何初始化策略：
- 隐藏层权重：$\mathcal{N}(0, \sqrt{2} / \sqrt{d_{\text{out}}})$
- 输出层权重：$\mathcal{N}(\sqrt{\pi} / \sqrt{d_{\text{in}}}, 10^{-5})$
- 输出层偏置：$-1.0$

该初始化使得网络初始状态的零等值面近似为一个单位球面，有助于加速收敛。

#### 损失函数

训练目标是最小化以下复合损失：

$$L = \lambda_{\text{sdf}} L_{\text{sdf}} + \lambda_{\text{grad}} L_{\text{grad}} + \lambda_{\text{eikonal}} L_{\text{eikonal}}$$

其中：
- SDF 损失：$L_{\text{sdf}} = \frac{1}{N} \sum_{i=1}^{N} (\tilde{F}(q_i) - F(q_i))^2$
- 梯度损失：$L_{\text{grad}} = \frac{1}{N} \sum_{i=1}^{N} \|\nabla \tilde{F}(q_i) - N(q_i)\|^2$
- Eikonal 损失：在 $[-0.5, 0.5]^3$ 内随机采样一半数量的点，约束 $\|\nabla \tilde{F}\| = 1$

### 1.2 训练策略

- 训练模式：surface（仅从表面点云采样）
- 采样数量：$25{,}000$ / iteration
- 迭代次数：$10{,}000$
- 优化器：Adam
- 学习率：$5 \times 10^{-4}$
- 学习率调度：CosineAnnealingLR
- $\lambda_{\text{sdf}}$：$1.0$
- $\lambda_{\text{grad}}$：$1.0$
- $\lambda_{\text{eikonal}}$：$0.1$

### 1.3 训练与重建时间

- 单 Shape 训练：约 $10$ 分钟 / $10{,}000$ iterations（CPU）
- $5$ 个 Shape 总训练时间：约 $50$ 分钟

### 1.4 重建结果与分析

我们对 ShapeNet V1 airplane 类别中的 $5$ 个样本进行了重建，并与 Ground Truth 进行了对比。

#### 总体观察

Base MLP 在 $10{,}000$ iterations 的 surface 模式下，能够恢复出飞机的大致几何轮廓，包括机身、机翼、尾翼等主要结构。但由于：
- 网络容量有限（$\text{hidden\_dim} = 512$，$10$ 层）
- 仅使用表面点云训练（缺乏空间采样点的 SDF 监督）
- ReLU 激活函数的固有平滑性偏好

重建结果呈现明显的过度平滑（over-smoothing）现象，高频细节大量丢失。

#### 样本 1：双翼飞机

| Ground Truth | 重建结果 |
|---|---|
| ![GT](data/1a04e3eab45ca15dd86060f189eb133.png) | ![重建](results/base/1a04e3eab45ca15dd86060f189eb133_surface.png) |

整体双翼结构被成功恢复，上下两层机翼的相对位置正确。但 GT 中清晰可见的螺旋桨细节完全消失，取而代之的是机身前端的平滑凸起。机翼的支撑杆（struts）也未能重建，上下翼之间呈现粘连状态。

#### 样本 2：喷气客机

| Ground Truth | 重建结果 |
|---|---|
| ![GT](data/1a32f10b20170883663e90eaf6b4ca52.png) | ![重建](results/base/1a32f10b20170883663e90eaf6b4ca52_surface.png) |

机身与机翼的整体布局重建良好，具有典型的客机流线型轮廓。但机翼末端的小翼（winglet）丢失，发动机短舱与机翼的连接处过度平滑。机身表面的微小起伏（如驾驶舱窗户轮廓）完全不可见。

#### 样本 3：四引擎螺旋桨飞机

| Ground Truth | 重建结果 |
|---|---|
| ![GT](data/1a54a2319e87bd4071d03b466c72ce41.png) | ![重建](results/base/1a54a2319e87bd4071d03b466c72ce41_surface.png) |

该样本的重建结果较好地保留了宽体机身与大展弦比机翼的结构特征。四个引擎的位置隐约可见（呈现为机翼下方的四个小凸起），但螺旋桨叶片完全丢失。机翼与机身的过渡区域比 GT 更圆滑，缺乏锐利的几何边界。

#### 样本 4：战斗机

| Ground Truth | 重建结果 |
|---|---|
| ![GT](data/1a6ad7a24bb89733f412783097373bdc.png) | ![重建](results/base/1a6ad7a24bb89733f412783097373bdc_surface.png) |

后掠翼布局被正确恢复，机身与垂直尾翼的结构关系合理。但 GT 中丰富的挂载细节（机翼下方的导弹/油箱）完全消失，起落架也没有被重建。此外，重建表面存在一些局部的不光滑区域，可能由表面点云法向噪声导致。

#### 样本 5：隐形战斗机

| Ground Truth | 重建结果 |
|---|---|
| ![GT](data/1a9b552befd6306cc8f2d5fe7449af61.png) | ![重建](results/base/1a9b552befd6306cc8f2d5fe7449af61_surface.png) |

该样本的重建效果相对较好，可能是由于隐形战斗机本身具有更简洁、更平滑的几何外形。飞翼式布局与菱形机身被较好地恢复。但进气道轮廓、起落架舱门等细节仍然丢失。

#### 关键问题总结

- 细节丢失严重：螺旋桨、起落架、武器挂载、翼梢小翼等高频几何特征完全无法重建
- 过度平滑：锐利的边缘（如机翼前缘、尾翼连接处）被圆角化处理
- 表面点云模式的局限：由于仅使用 surface 模式训练，缺乏空间 SDF 值监督，网络难以学习到远离表面的正确符号距离，导致整体几何精度受限

## 2. Fourier Feature MLP 表面重建

### 2.1 模型结构与方法

Fourier Feature MLP 在 Base MLP 的基础上，引入了高斯随机 Fourier 特征映射作为输入编码。其核心思想是将低维坐标映射到高维频域空间，使得 MLP 能够学习更高频的几何细节。

- 映射函数：$\gamma(q) = [\sin(2\pi B q), \cos(2\pi B q)]$，其中 $B \in \mathbb{R}^{3 \times M}$ 为随机高斯采样矩阵
- 映射维度：$M = 64$，实际输入维度为 $2M = 128$
- 高斯标准差（scale）：$\sigma = 5.0$
- 网络其余结构与 Base MLP 保持一致（$\text{hidden\_dim} = 512$，$10$ 层，ReLU，skip connection）

### 2.2 训练策略

- 训练模式：surface
- 采样数量：$25{,}000$ / iteration
- 迭代次数：$20{,}000$
- 优化器：Adam
- 学习率：$2 \times 10^{-4}$（Fourier 模式使用更低的学习率以稳定高频训练）
- 学习率调度：CosineAnnealingLR ($\eta_{\min} = 10^{-6}$)
- $\lambda_{\text{sdf}}$：$2.0$
- $\lambda_{\text{grad}}$：$0.5$
- $\lambda_{\text{eikonal}}$：$0.0$（禁用 Eikonal 损失以避免空白区域产生虚假零交叉）

### 2.3 训练与重建时间

- 单 Shape 训练：约 $22$ 分钟 / $20{,}000$ iterations（CPU）
- $5$ 个 Shape 总训练时间：约 $110$ 分钟

### 2.4 重建结果与分析

| Ground Truth | Base MLP | Fourier Feature MLP |
|---|---|---|
| ![GT](data/1a04e3eab45ca15dd86060f189eb133.png) | ![Base](results/base/1a04e3eab45ca15dd86060f189eb133_surface.png) | ![Fourier](results/fourier/1a04e3eab45ca15dd86060f189eb133_surface_fourier.png) |
| ![GT](data/1a32f10b20170883663e90eaf6b4ca52.png) | ![Base](results/base/1a32f10b20170883663e90eaf6b4ca52_surface.png) | ![Fourier](results/fourier/1a32f10b20170883663e90eaf6b4ca52_surface_fourier.png) |
| ![GT](data/1a54a2319e87bd4071d03b466c72ce41.png) | ![Base](results/base/1a54a2319e87bd4071d03b466c72ce41_surface.png) | ![Fourier](results/fourier/1a54a2319e87bd4071d03b466c72ce41_surface_fourier.png) |
| ![GT](data/1a6ad7a24bb89733f412783097373bdc.png) | ![Base](results/base/1a6ad7a24bb89733f412783097373bdc_surface.png) | ![Fourier](results/fourier/1a6ad7a24bb89733f412783097373bdc_surface_fourier.png) |
| ![GT](data/1a9b552befd6306cc8f2d5fe7449af61.png) | ![Base](results/base/1a9b552befd6306cc8f2d5fe7449af61_surface.png) | ![Fourier](results/fourier/1a9b552befd6306cc8f2d5fe7449af61_surface_fourier.png) |

相比 Base MLP，Fourier Feature MLP 在以下方面表现出明显的优势：

- 高频细节更丰富：双翼飞机的上下机翼之间出现了清晰的间隙，螺旋桨轮廓隐约可见；战斗机的机翼下方挂载物（导弹/油箱）的细节也有所恢复
- 边缘更锐利：机翼前缘、尾翼连接处等几何边界不再被过度圆角化
- 表面更贴合原始点云：整体几何形状更接近 Ground Truth，减少了 Base MLP 中的粘连和膨胀现象

然而，Fourier Feature 方法也带来了一些副作用：
- 空白区域存在少量漂浮面片（artifacts），这是高频网络在远离训练数据的区域容易产生微小零交叉的固有问题
- 通过 `--clean_mesh` 后处理（保留最大连通组件）可以有效去除这些漂浮面片

### 2.5 方法对比总结

| 对比维度 | Base MLP | Fourier Feature MLP |
|---|---|---|
| 低频轮廓恢复 | 良好 | 良好 |
| 高频细节（螺旋桨、挂载物等） | 大量丢失 | 明显恢复 |
| 边缘锐利度 | 过度平滑 | 更锐利 |
| 空白区域 artifacts | 较少 | 存在（可通过 clean_mesh 去除） |
| 单 Shape 训练时间 | $\sim 10$ min ($10$k iters) | $\sim 22$ min ($20$k iters) |

Fourier Feature 位置编码有效缓解了 Base MLP 的频谱偏置问题，使得网络能够重建更高频的表面细节，整体重建质量显著优于 Base MLP。
