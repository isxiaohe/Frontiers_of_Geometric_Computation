"""ModelNet40 PyTorch Dataset from HDF5 files."""
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

_HDF5_ROOT = Path("datasets/modelnet40_hdf5_2048")


# ─── Augmentation functions ────────────────────────────────────────────
# Each function takes (xyz, **kwargs) and returns augmented xyz.
# Point cloud xyz is assumed to be normalized to unit sphere before augmentation.

def aug_rotate_z(xyz):
    """Random rotation around z-axis."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return xyz @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def aug_rotate_xy(xyz):
    """Random rotation around a random axis in the xy-plane (perturbation)."""
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis[2] = 0  # keep in xy plane
    axis /= np.linalg.norm(axis) + 1e-8
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return xyz @ R.T


def aug_scale(xyz, scale_low=0.8, scale_high=1.25):
    """Random isotropic scaling."""
    scale = np.random.uniform(scale_low, scale_high)
    return xyz * scale


def aug_translate(xyz, shift_range=0.2):
    """Random translation."""
    shift = np.random.uniform(-shift_range, shift_range, size=3)
    return xyz + shift


def aug_jitter(xyz, sigma=0.01, clip=0.05):
    """Random Gaussian jitter per point."""
    return xyz + np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)


def aug_point_dropout(xyz, keep_ratio=0.875):
    """Randomly drop a fraction of points."""
    keep = int(xyz.shape[0] * keep_ratio)
    idx = np.random.choice(xyz.shape[0], keep, replace=False)
    return xyz[idx]


def aug_shuffle(xyz):
    """Randomly shuffle point order."""
    idx = np.random.permutation(xyz.shape[0])
    return xyz[idx]


# ─── Augmentation presets ──────────────────────────────────────────────

AUGMENT_PRESETS = {
    # PointNet++ style: z-rotation + jitter + dropout
    'pointnet2': ['rotate_z', 'jitter', 'point_dropout'],
    # PointNeXt style: no rotation, scale + translate + jitter (modern recipe)
    'pointnext': ['scale', 'translate', 'jitter'],
    # Full augmentation: all transforms
    'full': ['rotate_z', 'scale', 'translate', 'jitter', 'point_dropout'],
    # None
    'none': [],
}

# Default kwargs per augmentation
AUGMENT_DEFAULTS = {
    'rotate_z': {},
    'rotate_xy': {},
    'scale': {'scale_low': 0.8, 'scale_high': 1.25},
    'translate': {'shift_range': 0.2},
    'jitter': {'sigma': 0.01, 'clip': 0.05},
    'point_dropout': {'keep_ratio': 0.875},
    'shuffle': {},
}

# Map names to functions
AUGMENT_FNS = {
    'rotate_z': aug_rotate_z,
    'rotate_xy': aug_rotate_xy,
    'scale': aug_scale,
    'translate': aug_translate,
    'jitter': aug_jitter,
    'point_dropout': aug_point_dropout,
    'shuffle': aug_shuffle,
}


# ─── Dataset ───────────────────────────────────────────────────────────

class ModelNet40H5(Dataset):
    """ModelNet40 from pre-sampled HDF5 files (2048 points per shape).

    Expects train.h5 and test.h5 with:
        data:  (N, 2048, 3)  float32
        label: (N, 1)        int64

    Args:
        root:       path to HDF5 directory.
        split:      'train' or 'test'.
        num_points: number of points to sample.
        normalize:  normalize to unit sphere.
        augment:    augmentation preset name (str) or list of augment names,
                    or False for no augmentation.
    """

    def __init__(self, root=_HDF5_ROOT, split="train", num_points=1024,
                 normalize=True, augment=False):
        super().__init__()
        self.num_points = num_points
        self.normalize = normalize

        # Resolve augment list
        if augment is False or augment == 'none':
            self.augment_list = []
        elif isinstance(augment, str):
            # Preset name
            if augment in AUGMENT_PRESETS:
                self.augment_list = AUGMENT_PRESETS[augment]
            else:
                raise ValueError(f"Unknown augment preset: {augment}. "
                                 f"Available: {list(AUGMENT_PRESETS.keys())}")
        elif isinstance(augment, list):
            self.augment_list = augment
        else:
            self.augment_list = []

        import h5py

        h5_path = Path(root) / f"{split}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            self.data = f["data"][:]                      # (N, 2048, 3)
            self.label = f["label"][:].flatten()           # (N,)

        aug_str = ', '.join(self.augment_list) if self.augment_list else 'none'
        print(f"[ModelNet40H5] split={split}, samples={len(self.label)}, "
              f"num_points={num_points}, augment=[{aug_str}]")

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
        for aug_name in self.augment_list:
            fn = AUGMENT_FNS[aug_name]
            kwargs = AUGMENT_DEFAULTS.get(aug_name, {})
            xyz = fn(xyz, **kwargs)

        return xyz.astype(np.float32), label


def build_dataloader(config):
    """Build train/test DataLoaders from a config dict.

    Reads 'augment' from config (preset name or list). Defaults to 'pointnet2'.
    """
    root = config.get("data_root", str(_HDF5_ROOT))
    num_points = config["num_points"]
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    augment = config.get("augment", "pointnet2")

    train_ds = ModelNet40H5(root, "train", num_points, normalize=True, augment=augment)
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
