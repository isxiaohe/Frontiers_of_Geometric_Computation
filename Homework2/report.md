# 几何计算前沿 第二次作业

## 项目介绍

使用 Python 实现了 [QEM 网格简化算法](https://dl.acm.org/doi/10.1145/258734.258849)（Garland & Heckbert, 1997）。

基本使用方法：

```bash
pip install numpy
python mesh_simplifier.py --input <input.obj> --output <output.obj> --scale <ratio>
```

```
options:
  -h, --help       show this help message and exit
  --input INPUT    Path to the input mesh file (OBJ format)
  --output OUTPUT  Path to the output mesh file (OBJ format)
  --scale SCALE    Simplification ratio in (0, 1]. Target face count = original faces * scale.

examples:
  python mesh_simplifier.py --input bunny.obj --output bunny_lo.obj --scale 0.5
  python mesh_simplifier.py --input input.obj --output output.obj --scale 0.3
```

## 实现说明

### Mesh 数据结构

使用**索引面片（Indexed Face Set）** 表示三角网格：

- `vertices`：`numpy` 二维数组 `(N, 3)`，存储所有顶点的三维坐标。
- `faces`：列表，每个元素为三个顶点索引 `[v0, v1, v2]`，表示一个三角面。
- `vertex_faces`：`defaultdict(set)`，顶点到其相邻面索引的映射，用于快速查询邻接关系。

此外使用 `deleted_vertices` 和 `deleted_faces` 两个集合标记被删除的元素，避免频繁重建数组。边的表示采用 `(min(v1, v2), max(v1, v2))`，确保无向边的唯一性。

### 算法流程

```
Input: 三角网格 M = (V, F)，目标面数 T
Output: 简化后的三角网格 M'

1. 计算每个面的法向量 n_f 和平面方程 (n, d)
2. 计算每个面的二次误差矩阵 Q_f = [n; d] [n; d]^T
3. 计算每个顶点的二次误差矩阵 Q_v = Σ_{f ∈ star(v)} Q_f
4. 对每条边 e = (v1, v2):
   a. 计算合并代价 cost = x^T A x + 2b^T x + c，其中 Q = Q_{v1} + Q_{v2}
   b. 求最优合并位置 x*
   c. 将 (cost, e) 插入最小堆
5. while 当前面数 > T:
   a. 从最小堆中取出代价最小的边 e = (v1, v2)
   b. 检验边坍缩的合法性
   c. 若合法，将 v1、v2 合并到最优位置 x*，删除共享面
   d. 更新受影响区域的法向量、二次误差矩阵和优先队列
6. 返回简化后的网格
```

### QEM 计算

对于边 `(v1, v2)`，合并后的误差度量定义为：

$$E(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + 2\mathbf{b}^T \mathbf{x} + c = [\mathbf{x}^T, 1]\, Q\, [\mathbf{x}; 1]$$

其中 $Q = Q_{v1} + Q_{v2}$，$A = Q_{[:3,:3]}$，$\mathbf{b} = Q_{[:3,3]}$，$c = Q_{[3,3]}$。

最优合并位置 $\mathbf{x}^*$ 应使 $E(\mathbf{x})$ 最小，对 $\mathbf{x}$ 求导令其为零：

$$\frac{\partial E}{\partial \mathbf{x}} = 2A\mathbf{x} + 2\mathbf{b} = 0 \implies \mathbf{x}^* = -A^{-1}\mathbf{b}$$

然而 $A$ 可能病态或奇异（如平坦区域），直接求逆会导致数值不稳定。因此使用 **Tikhonov 正则化**方法求解：

$$(A + \lambda I)\,\mathbf{x} = -\mathbf{b}$$

其中 $\lambda = 10^{-6} \cdot \max(\text{diag}(A))$，自适应地与 $A$ 的尺度匹配。当 $A$ 状态良好时，$\lambda$ 极小，解接近精确解；当 $A$ 病态时，正则化项保证系统始终可解且结果稳定。

此外，当正则化解的代价比 $\mathbf{v}_1$、$\mathbf{v}_2$ 或中点 $\frac{\mathbf{v}_1+\mathbf{v}_2}{2}$ 更差时，自动回退为选取这四个候选点中代价最小的位置，确保不会因数值问题产生不合理的结果。

## 结果

分别以 `scale = 0.5`、`0.1`、`0.01` 对原始模型进行简化：

| scale = 0.5（保留 50%） | scale = 0.1（保留 10%） | scale = 0.01（保留 1%） |
|:---:|:---:|:---:|
| ![](outputs/scale_0_5.png) | ![](outputs/scale_0_1.png) | ![](outputs/scale_0_01.png) |

可以看到：
- **scale = 0.5**：模型外形与细节保持良好，整体轮廓几乎无变化。
- **scale = 0.1**：细节（如角、爪）开始丢失，但主体姿态仍然可辨认。
- **scale = 0.01**：面数极少，仅保留最基本的轮廓，大量细节消失。


