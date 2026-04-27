# 几何计算前沿 第一次作业
### 杨知非 2300012416

## 运行方式

```bash
python demo.py [--color]
```

## Task1: 从深度图生成点云

将深度图像素坐标 $(u, v)$ 转换到世界坐标系：

$$
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad
P_{world} = T_{cam2world} \cdot P_{cam}
$$

```python
    img_to_cam = np.linalg.inv(cam_intr)
    x, y = np.meshgrid(np.arange(depth_im.shape[0]), np.arange(depth_im.shape[1]), indexing='ij')
    z = depth_im.flatten()
    cam_pts = img_to_cam @ np.vstack((y * z, x * z, z))
    world_pts = (cam_pose @ np.vstack((cam_pts, np.ones((1, cam_pts.shape[1])))))[:3, :].T
```

根据所有帧点云确定体素场边界：
```python
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.min(view_frust_pts, axis=0))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.max(view_frust_pts, axis=0))
```

**点云可视化**（每隔100帧导出）：
![Point Cloud](results/pcd_from_10_rad_frame.png)

## Task2: 体素投影与深度采样

将体素中心依次转换：体素坐标 → 世界坐标 → 相机坐标 → 像素坐标

```python
    vox_world = self.vox_coords * self.voxel_size + self.vol_bnds[:, 0]
    vox_cam = (np.linalg.inv(cam_pose) @ np.hstack((vox_world, np.ones((len(vox_world), 1)))).T).T[:, :3]
    pix_z = vox_cam[:, 2]
    pix = np.round((cam_intr @ vox_cam.T).T[:, :2] / pix_z[:, None])
    depth_val = depth_im[pix[:, 1].astype(int), pix[:, 0].astype(int)]
```

## Task3: 计算单帧TSDF

$$
TSDF(x) = \text{clip}\left(\frac{depth - z}{t}, -1, 1\right)
$$

其中 $t$ 为截断距离（取 $5 \times voxel\_size$）：

```python
    depth_diff = depth_val - pix_z
    valid_pts = depth_diff >= -self.trunc_margin
    tsdf = np.clip(depth_diff / self.trunc_margin, -1, 1)
```

## Task4: 多帧TSDF融合

加权平均融合：

$$
D_{i+1}(x) = \frac{W_i(x)D_i(x) + w_{i+1}d_{i+1}(x)}{W_i(x) + w_{i+1}}
$$

```python
    valid_pix_idx = np.where(valid_pix)[0]
    final_valid_idx = valid_pix_idx[valid_pts]
    w_old = self.weight_vol[final_valid_idx]
    w_new = w_old + obs_weight
    self.tsdf_vol[final_valid_idx] = (self.tsdf_vol[final_valid_idx] * w_old + tsdf * obs_weight) / w_new
    self.weight_vol[final_valid_idx] = w_new
```

**Mesh提取**：使用 Marching Cubes 在 TSDF=0 等值面上提取三角网格。

**重建结果**：

| 灰度Mesh | 彩色Mesh |
|:--------:|:--------:|
| ![Gray Mesh](results/mesh.png) | ![Color Mesh](results/color_mesh.png) |

## Bonus: 颜色融合

颜色同样采用加权平均融合：

```python
    color_val = color_im[pix[:, 1].astype(int), pix[:, 0].astype(int)] / 255.0
    self.color_vol[final_valid_idx] = (self.color_vol[final_valid_idx] * w_old[:, None] + color_val * obs_weight) / w_new[:, None]
```

顶点颜色通过最近邻插值从 color_vol 获取：

```python
    verts_int = verts.astype(int)
    verts_idx = (verts_int[:, 0] * vol_dims[1] * vol_dims[2] +
                 verts_int[:, 1] * vol_dims[2] + verts_int[:, 2])
    vertex_colors = (tsdf_vol.color_vol[verts_idx] * 255).clip(0, 255).astype(np.uint8)
```
