"""ModelNet40 PyTorch Dataset from HDF5 files."""
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

_HDF5_ROOT = Path("datasets/modelnet40_hdf5_2048")


class ModelNet40H5(Dataset):
    """ModelNet40 from pre-sampled HDF5 files (2048 points per shape).

    Expects train.h5 and test.h5 with:
        data:  (N, 2048, 3)  float32
        label: (N, 1)        int64
    """

    def __init__(self, root=_HDF5_ROOT, split="train", num_points=1024,
                 normalize=True, augment=False):
        super().__init__()
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment

        import h5py

        h5_path = Path(root) / f"{split}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            self.data = f["data"][:]                      # (N, 2048, 3)
            self.label = f["label"][:].flatten()           # (N,)

        print(f"[ModelNet40H5] split={split}, samples={len(self.label)}, "
              f"num_points={num_points}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        xyz = self.data[idx].copy()  # (2048, 3)
        label = int(self.label[idx])

        # Random downsample to num_points
        if xyz.shape[0] > self.num_points:
            choice = np.random.choice(xyz.shape[0], self.num_points, replace=False)
            xyz = xyz[choice]

        # Normalize to unit sphere
        if self.normalize:
            centroid = xyz.mean(axis=0)
            xyz -= centroid
            max_dist = np.linalg.norm(xyz, axis=1).max()
            if max_dist > 0:
                xyz /= max_dist

        # Augmentation (train only)
        if self.augment:
            xyz = self._augment(xyz)

        return xyz.astype(np.float32), label

    def _augment(self, xyz):
        # Random z-axis rotation
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        xyz = xyz @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

        # Random jitter
        xyz += np.clip(0.01 * np.random.randn(*xyz.shape), -0.05, 0.05)

        # Random point dropout (keep 87.5%)
        keep = int(xyz.shape[0] * 0.875)
        idx = np.random.choice(xyz.shape[0], keep, replace=False)
        xyz = xyz[idx]

        return xyz


def build_dataloader(config):
    """Build train/test DataLoaders from a config dict."""
    root = config.get("data_root", str(_HDF5_ROOT))
    num_points = config["num_points"]
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)

    train_ds = ModelNet40H5(root, "train", num_points, normalize=True, augment=True)
    test_ds = ModelNet40H5(root, "test", num_points, normalize=True, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
