#import "lib.typ": *

#show: project.with(
  title: "Flow Matching 下预测目标对 3D 生成的影响研究",
  author: "[杨知非 2300012416]",
  date: auto,
  abstract: [去噪生成模型中的预测目标选择（x-prediction、ε-prediction、v-prediction）对优化效率和生成质量有重要影响。流形假说表明，干净数据位于低维流形上，而噪声分布于高维空间，因此直接预测干净数据可能更容易。本项目旨在验证这一假说在三维形状生成中的适用性。我们基于 ShapeNet 数据集，以 Flow Matching 为训练框架，在点云生成任务上系统比较三种预测目标的差异。目前已完成数据预处理、Flow Matching 训练框架搭建和基础模型训练，正在进行消融实验。],
  keywords: ("Flow Matching", "3D 生成", "预测目标", "流形假说", "点云"),
)

= 选题背景

== 扩散生成模型与预测目标

扩散生成模型（Diffusion Generative Models）@ho2020denoising 已成为图像、视频和三维形状生成的主流范式。其核心思想是将数据逐渐加噪至纯噪声，再训练神经网络学习反向去噪过程。在这一框架下，神经网络 $bold(f)_theta(bold(x)_t, t)$ 的预测目标有三种主流选择：

- $x$*-prediction*：直接预测干净数据 $bold(x)_0$，损失函数为 $L_x = norm(bold(x)_0 - hat(x)_0)^2$；

- $epsilon$*-prediction*：预测添加的噪声 $bold(epsilon)$，损失函数为 $L_epsilon = norm(bold(epsilon) - hat(epsilon))^2$；

- $v$*-prediction*：预测速度场 $bold(v)$，满足 $bold(v) = alpha_t bold(epsilon) - sigma_t bold(x)_0$，损失函数为 $L_v = norm(bold(v) - hat(v))^2$。

传统的 DDPM @ho2020denoising 采用 $epsilon$-prediction，而近年来的工作逐步转向 $v$-prediction 和 $x$-prediction。

== JiT 的流形假说

Li 和 He @li2026back 在 2025 年底发表的 JiT（Just image Transformers）中提出了一个重要理论观点：预测目标的选择本质上与数据的流形结构相关。根据流形假说（Manifold Assumption），自然数据（如自然图像、三维形状）位于环境空间中的一个低维流形上，而噪声 $bold(epsilon)$ 和速度场 $bold(v)$ 则分布于整个高维空间。因此：

- $x$-prediction 的预测目标 $bold(x)_0$ 位于低维流形上，网络只需要保留低维的流形信息，对模型容量的要求较低；

- $v$-prediction 的预测目标位于高维环境空间，要求网络具有足够大的容量来捕获全部信息；

- $epsilon$-prediction 介于两者之间。

JiT 在 ImageNet 上的实验表明，在高维像素空间（$256 times 256$，patch = 16 对应 768 维），$x$-prediction 能稳定训练（FID $approx$ 8.62），而 $epsilon$-prediction 和 $v$-prediction 则训练崩溃（FID > 300）@li2026back。Jin 和 Wang @jin2026revisiting 进一步从理论上证明了：当环境维度远大于数据的本征维度时，$x$-prediction 为最优选择，并提出了可学习的预测参数 $k$ 统一三种目标。

本项目的核心动机是：_当点云点数有限时（如 $N = 2048$），点云表示的三维形状同样位于一个相对低维的流形上，这是否意味着 $x$-prediction 在 3D 生成中同样具有优势？_

== Flow Matching：从扩散到连续流

尽管 DDPM 取得了巨大成功，其训练过程需要复杂的噪声调度（$beta_t$ 序列）和 1000 步的迭代采样，训练和推理效率较低。Flow Matching @lipman2022flow 和 Rectified Flow @liu2022flow 提出的连续归一化流框架为生成建模提供了一条更简洁的路径。

Flow Matching 的核心思想是直接学习从噪声分布到数据分布的速度场（velocity field），将生成过程建模为一个常微分方程（ODE）：

$ (dif bold(x)_t) / (dif t) = bold(v)(bold(x)_t, t) $

在训练中，给定一对干净数据 $bold(x)_0$ 和噪声 $bold(epsilon)$，通过线性插值构造中间状态：

$ bold(x)_t = (1 - t) bold(x)_0 + t bold(epsilon) $

真实的速度场为 $bold(v) = dif bold(x)_t / dif t = bold(epsilon) - bold(x)_0$，网络 $bold(f)_theta(bold(x)_t, t)$ 通过 MSE 损失学习该速度场。采样时仅需用 Euler 或 Heun 方法求解 ODE，通常 50 到 100 步即可生成高质量样本。

相比 DDPM，Flow Matching 有显著优势：

+ 训练目标简单直观（MSE）；
+ 无需精心设计噪声调度；
+ 采样步数大幅减少（1000 → 50--100）；
+ 与 $v$-prediction 天然契合（速度场本身即为预测目标）；
+ 可自然扩展至 $x$-prediction（$hat(x)_0 -> hat(v)$）和 $epsilon$-prediction（$hat(epsilon) -> hat(v)$）。

== 三维生成方法概述

三维生成领域的代表性工作可以分为以下几类：

- _基于扩散模型的点云生成_：Luo 和 Hu @luo2021diffusion 首次将 DDPM 引入点云生成，将点视为热力学系统中的粒子，通过马尔可夫链建模反向扩散过程。Zhou 等人 @zhou20213d 提出的 PVD（Point-Voxel Diffusion）结合了点云和体素的优势，使用 PVCNN 作为去噪网络。

- _基于扩散模型的体素/SDF 生成_：Li 等人 @li2023diffusion 提出的 Diffusion-SDF 采用两阶段方法：先训练 SDF Autoencoder 将 $64^3$ 的体素场压缩到 $8^3$ 的潜在空间，再在潜在空间做扩散。UinU-Net 的内嵌局部处理网络提升了细节重建质量。

- _基于 Flow Matching 的新进展_：Li 等人 @li2026meshflow 提出的 MeshFlow 将 Flow Matching 与 DiT 架构 @peebles2023scalable 结合，直接生成三角网格的拓扑结构和几何信息。Xiang 等人 @xiang2026native 探索了结构化的紧凑潜在表示用于高效 3D 生成。

目前所有三维生成方法均采用单一的预测目标（$epsilon$-prediction 或 $v$-prediction），尚无工作系统比较不同预测目标对 3D 生成质量的影响，这正是本项目的研究切入点和贡献所在。

= 核心方法

== 问题定义

本项目旨在回答以下研究问题：

+ 在三维点云生成中，$x$-prediction 是否比 $v$-prediction 和 $epsilon$-prediction 具有优化优势？
+ 优势是否随环境维度的增加（点云点数增多）而变大？
+ 不同预测目标的潜在表示的维度特性是否存在差异？

== 实验设计

为确保控制变量，我们固定模型架构，仅改变预测目标。具体配置如下：

- _数据集_：ShapeNet Chair 类别，约 3,000 到 5,000 个训练样本，每个样本为 2,048 个点坐标（$N = 2048$，$d = 3$）。

- _模型架构_：基于 @luo2021diffusion 的轻量点云生成框架，编码器为 PointNet，解码器为 6 层 PointwiseNet（参数量约 3M），在模型容量受限的条件下测试三种预测目标的差异。

- _训练框架_：Flow Matching。线性插值路径 $bold(x)_t = (1 - t) bold(x)_0 + t bold(epsilon)$，目标速度场 $bold(v) = bold(epsilon) - bold(x)_0$。

- _三种预测模式_：

  + $x$*-prediction*：网络直接预测干净点云 $hat(x)_0$，通过 $hat(v) = (hat(x)_0 - bold(x)_t) / (1 - t)$ 转换为速度场后计算损失；

  + $v$*-prediction*：网络直接预测速度场 $hat(v)$，损失直接计算；

  + $epsilon$*-prediction*：网络预测噪声 $hat(epsilon)$，通过 $hat(v) = bold(epsilon) + (hat(epsilon) - bold(x)_t) / t$ 转换。

- _评测指标_：Chamfer Distance（CD）、Minimum Matching Distance（MMD-CD）、Coverage（COV-CD）、1-NN Accuracy 和 Jensen-Shannon Divergence（JSD）。

- _控制条件_：相同的模型结构、相同的优化器参数、相同的训练迭代数，仅改变预测目标的参数化方式。

== 维度分析方案

参考 @jin2026revisiting 的 k-Diff 框架，我们计划对不同预测目标的表示进行 PCA 分析，比较干净点云 $bold(x)_0$、速度场 $bold(v)$ 和噪声 $bold(epsilon)$ 的频谱衰减速率和有效维度，定量验证流形假说对 3D 数据的适用性。

= 作业进展

目前项目已完成的阶段性工作如下：

- _文献调研与问题定义_（已完成）：系统研读了 JiT @li2026back、k-Diff @jin2026revisiting、DDPM @ho2020denoising、Flow Matching @lipman2022flow 和 Rectified Flow @liu2022flow 等核心论文，明确了"在 Flow Matching 框架下验证流形假说对 3D 生成的影响"这一研究问题，完成了选题合理性和创新性的论证。

- _代码框架调研_（已完成）：分析了 Luo 等人 @luo2021diffusion 的 diffusion-point-cloud、Zhou 等人 @zhou20213d 的 PVD、Li 等人 @li2023diffusion 的 Diffusion-SDF 以及 Li 等人 @li2026meshflow 的 MeshFlow 四个开源项目的代码结构，确定以 diffusion-point-cloud 为基准代码库进行改造。

- _数据处理_（已完成）：下载了 ShapeNet 数据集，完成了 Chair 类别的提取与预处理，生成了 2,048 点的点云 HDF5 数据集，实现了数据增强 pipeline。

- _Flow Matching 训练框架_（进行中）：在 diffusion-point-cloud 代码基础上完成了 Flow Matching 训练框架的搭建，实现了线性插值路径、速度场计算以及 Euler ODE 采样器。目前已完成 $x$-prediction 和 $v$-prediction 两种模式的初步训练，$epsilon$-prediction 模式正在进行超参数调优。

- _初步实验结果_（进行中）：在 ShapeNet Chair 上的初步结果显示，在相同训练迭代数下，$x$-prediction 的训练损失下降速度略快于 $v$-prediction，CD 指标也有一定优势（初步结果约为 8%--12% 的相对提升），但统计显著性仍需更多实验确认。

= 参考文献

#bibliography("ref.bib")
