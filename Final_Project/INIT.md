下面这个版本比较适合作为课程项目 proposal 的初稿。

Do JiT’s Manifold Arguments Hold for 3D Diffusion Models?

Motivation

Recent work Back to Basics: Let Denoising Generative Models Denoise (JiT) argues that the choice of diffusion prediction target plays a fundamental role in optimization. The paper proposes that natural data lie on a low-dimensional manifold, while noise occupies the full ambient space. As a result, directly predicting clean data (x_0) may be substantially easier than predicting noise (\epsilon) or velocity (v), especially when model capacity is limited.

While this phenomenon has been studied on image generation benchmarks, it remains unclear whether the same conclusions hold for 3D shape generation.

3D objects from ShapeNet are also believed to lie on extremely low-dimensional manifolds. For example, although a voxelized chair may have tens of thousands of dimensions, the actual degrees of freedom governing chair geometry are much smaller. Therefore, 3D shape generation provides an interesting testbed for evaluating JiT’s manifold hypothesis.

⸻

Research Question

Does x-prediction provide optimization advantages over v-prediction and ε-prediction in 3D diffusion models?

More specifically:

* Does x-pred converge faster?
* Does x-pred achieve better generation quality under limited model capacity?
* Does the advantage increase when the data representation becomes more low-dimensional and structured?

⸻

Method

Dataset

ShapeNet

Initially focus on a single category:

* Chair

using approximately 1,000–3,000 training samples.

⸻

Representation

Primary representation:

* Occupancy voxel grids
* Resolution: 32^3

Optional extension:

* Signed Distance Fields (SDF)

The occupancy representation is particularly interesting because it forms a highly structured low-dimensional manifold, which should amplify the effects predicted by JiT.

⸻

Diffusion Model

Use a deliberately simple backbone:

* Tiny 3D U-Net
* 2–5M parameters
* Standard DDPM noise schedule

The backbone remains fixed across all experiments.

Only the prediction target changes.

Input:

(x_t, t)

Output:

f_\theta(x_t,t)

Three variants:

x-prediction

f_\theta(x_t,t) \rightarrow x_0

Loss:

L_x = \|x_0 - \hat{x}_0\|^2

ε-prediction

f_\theta(x_t,t) \rightarrow \epsilon

Loss:

L_\epsilon = \|\epsilon - \hat{\epsilon}\|^2

v-prediction

f_\theta(x_t,t) \rightarrow v

where

v = \alpha_t \epsilon - \sigma_t x_0

Loss:

L_v = \|v - \hat{v}\|^2

⸻

Evaluation

Optimization

Track:

* Training loss
* Validation loss
* Convergence speed

⸻

Generation Quality

Potential metrics:

* Chamfer Distance (CD)
* MMD-CD
* Coverage

depending on the final representation.

⸻

Timestep-wise Analysis

Measure:

L_t

for different diffusion timesteps.

This helps identify where x-pred or v-pred gains arise.

⸻

JiT-style Analysis

Beyond generation quality, investigate the geometry of prediction targets.

Perform PCA on:

x_0

and

v

representations.

Compare spectrum decay rates and effective dimensions.

Hypothesis:

* x_0 occupies a much lower-dimensional manifold.
* v is closer to a high-dimensional Gaussian distribution.
* This difference explains optimization behavior.

⸻

Optional Extension

Study representation complexity:

Representation	Ambient Dimension
16³ Occupancy	4,096
32³ Occupancy	32,768
64³ Occupancy	262,144

Question:

Does the performance gap between x-pred and v-pred grow as dimensionality increases?

This directly mirrors JiT’s large-patch experiments in the 3D setting.

⸻

Expected Outcome

If JiT’s hypothesis transfers to 3D generation, we expect:

* x-prediction to converge faster than ε-prediction and v-prediction.
* The advantage to become more pronounced on occupancy representations.
* The gap to increase as ambient dimensionality grows.
* PCA analysis to reveal significantly lower intrinsic dimensionality for x_0 compared to v.

These results would provide evidence that the manifold-based explanation proposed by JiT extends beyond image generation and also applies to 3D shape generation.

主要参考论文

1. Back to Basics: Let Denoising Generative Models Denoise (JiT, 2025)
    * 你的核心理论来源。
    * 研究 prediction target（x/ε/v）与数据流形之间的关系。
    * 建议精读全文。
2. Diffusion Probabilistic Models for 3D Point Cloud Generation (CVPR 2021)
    * 最早的 3D diffusion 工作之一。
    * 提供最简单的 3D diffusion 实现范式。
    * 很适合理解训练流程。
3. 3D Shape Generation and Completion through Point-Voxel Diffusion (PVD, ICCV 2021)
    * ShapeNet + voxel diffusion 的经典工作。
    * 可以参考数据预处理和评价指标。
4. Diffusion-SDF: Text-to-Shape via Voxelized Diffusion (2022)
    * 直接对 SDF/voxel 场做 diffusion。
    * 可参考 3D U-Net 设计。
5. Revisiting Diffusion Model Predictions Through Dimensionality (2026)
    * 可以看作 JiT 思想的后续理论分析。
    * 重点讨论 prediction target 与 intrinsic dimension 的关系。

如果最后真的落地，我会建议你把项目规模控制在：

ShapeNet Chair
32³ Occupancy
Tiny 3D UNet
x / ε / v prediction

先跑出第一版结果。这个配置在一张 A100 80GB 上非常轻松，而且最符合“控制变量、验证理论”的目标。