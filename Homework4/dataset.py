import os
import numpy as np
import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    """Iteration-based dataset with mixed sampling support.

    Modes:
      - "sdf": sample only from sdf.npz
      - "surface": sample only from pointcloud.npz (sdf=0, grad=normal)
      - "mixed": mix sdf and surface points by mix_ratio, then shuffle
    """

    def __init__(
        self,
        data_dir,
        mode="mixed",
        sample_size=25000,
        mix_ratio=0.5,
        num_iters=100000,
    ):
        super().__init__()
        assert mode in ("sdf", "surface", "mixed")
        self.mode = mode
        self.sample_size = sample_size
        self.mix_ratio = mix_ratio
        self.num_iters = num_iters

        sdf_path = os.path.join(data_dir, "sdf.npz")
        pc_path = os.path.join(data_dir, "pointcloud.npz")

        sdf_data = np.load(sdf_path)
        self.sdf_points = torch.from_numpy(sdf_data["points"]).float()
        self.sdf_grad = torch.from_numpy(sdf_data["grad"]).float()
        self.sdf_values = torch.from_numpy(sdf_data["sdf"]).float().unsqueeze(-1)

        pc_data = np.load(pc_path)
        self.surface_points = torch.from_numpy(pc_data["points"]).float()
        self.surface_normals = torch.from_numpy(pc_data["normals"]).float()

        all_points = torch.cat([self.sdf_points, self.surface_points], dim=0)
        self.bbox_min = all_points.min(dim=0)[0].numpy()
        self.bbox_max = all_points.max(dim=0)[0].numpy()

    def __len__(self):
        return self.num_iters

    def __getitem__(self, idx):
        if self.mode == "sdf":
            idxs = torch.randint(0, self.sdf_points.shape[0], (self.sample_size,))
            points = self.sdf_points[idxs]
            grad = self.sdf_grad[idxs]
            sdf = self.sdf_values[idxs]
        elif self.mode == "surface":
            idxs = torch.randint(0, self.surface_points.shape[0], (self.sample_size,))
            points = self.surface_points[idxs]
            grad = self.surface_normals[idxs]
            sdf = torch.zeros(self.sample_size, 1, dtype=torch.float32)
        else:  # mixed
            n_sdf = int(self.sample_size * self.mix_ratio)
            n_surface = self.sample_size - n_sdf

            idx_sdf = torch.randint(0, self.sdf_points.shape[0], (n_sdf,))
            idx_surface = torch.randint(0, self.surface_points.shape[0], (n_surface,))

            points = torch.cat(
                [self.sdf_points[idx_sdf], self.surface_points[idx_surface]], dim=0
            )
            grad = torch.cat(
                [self.sdf_grad[idx_sdf], self.surface_normals[idx_surface]], dim=0
            )
            sdf = torch.cat(
                [self.sdf_values[idx_sdf], torch.zeros(n_surface, 1, dtype=torch.float32)],
                dim=0,
            )

            perm = torch.randperm(self.sample_size)
            points = points[perm]
            grad = grad[perm]
            sdf = sdf[perm]

        return {
            "points": points,
            "grad": grad,
            "sdf": sdf,
        }
