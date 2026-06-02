import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):
    """加载单个shape的SDF采样数据为PyTorch Dataset。"""

    def __init__(self, data_dir, use_surface_points=False):
        """
        Args:
            data_dir: shape对应的文件夹路径，如 data/<uid>
            use_surface_points: 是否将表面点云（sdf=0）也加入训练集
        """
        super().__init__()
        sdf_path = os.path.join(data_dir, "sdf.npz")
        pc_path = os.path.join(data_dir, "pointcloud.npz")

        sdf_data = np.load(sdf_path)
        self.points = torch.from_numpy(sdf_data["points"]).float()
        self.grad = torch.from_numpy(sdf_data["grad"]).float()
        self.sdf = torch.from_numpy(sdf_data["sdf"]).float().unsqueeze(-1)

        if use_surface_points:
            pc_data = np.load(pc_path)
            surface_pts = torch.from_numpy(pc_data["points"]).float()
            surface_normals = torch.from_numpy(pc_data["normals"]).float()
            surface_sdf = torch.zeros(surface_pts.shape[0], 1, dtype=torch.float32)

            self.points = torch.cat([self.points, surface_pts], dim=0)
            self.grad = torch.cat([self.grad, surface_normals], dim=0)
            self.sdf = torch.cat([self.sdf, surface_sdf], dim=0)

        # 计算bbox用于测试时确定网格范围
        self.bbox_min = self.points.min(dim=0)[0].numpy()
        self.bbox_max = self.points.max(dim=0)[0].numpy()

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return {
            "points": self.points[idx],
            "grad": self.grad[idx],
            "sdf": self.sdf[idx],
        }
