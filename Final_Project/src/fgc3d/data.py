"""Small deterministic synthetic point-cloud dataset."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SHAPENET_CATEGORY_TO_SYNSET = {
    "airplane": "02691156",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bottle": "02876657",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "can": "02747177",
    "camera": "02942699",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "clock": "03046257",
    "dishwasher": "03207941",
    "monitor": "03211117",
    "table": "04379243",
    "telephone": "04401088",
    "tin_can": "02946921",
    "tower": "04460130",
    "train": "04468005",
    "keyboard": "03085013",
    "earphone": "03261776",
    "faucet": "03325088",
    "file": "03337140",
    "guitar": "03467517",
    "helmet": "03513137",
    "jar": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "speaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "microwave": "03761084",
    "motorcycle": "03790512",
    "mug": "03797390",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "pot": "03991062",
    "printer": "04004475",
    "remote_control": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "skateboard": "04225987",
    "sofa": "04256520",
    "stove": "04330267",
    "vessel": "04530566",
    "washer": "04554684",
    "cellphone": "02992529",
    "birdhouse": "02843684",
    "bookshelf": "02871439",
}


class SyntheticPointCloudDataset(Dataset[torch.Tensor]):
    """Generate simple chair-like point clouds without external data files."""

    def __init__(self, num_shapes: int = 128, num_points: int = 64, seed: int = 0) -> None:
        if num_shapes <= 0:
            raise ValueError("num_shapes must be positive")
        if num_points < 8:
            raise ValueError("num_points must be at least 8")
        self.num_shapes = num_shapes
        self.num_points = num_points
        self.seed = seed

    def __len__(self) -> int:
        return self.num_shapes

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.num_shapes:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)

        points = torch.empty(self.num_points, 3)
        seat_count = self.num_points // 2
        back_count = self.num_points // 4
        leg_count = self.num_points - seat_count - back_count

        seat_xy = torch.rand(seat_count, 2, generator=generator) * 2.0 - 1.0
        seat_z = torch.zeros(seat_count, 1)
        points[:seat_count] = torch.cat([seat_xy, seat_z], dim=1)

        back_x = torch.rand(back_count, 1, generator=generator) * 2.0 - 1.0
        back_y = torch.ones(back_count, 1)
        back_z = torch.rand(back_count, 1, generator=generator) * 1.4
        points[seat_count : seat_count + back_count] = torch.cat([back_x, back_y, back_z], dim=1)

        corners = torch.tensor(
            [[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]],
            dtype=torch.float32,
        )
        chosen = corners[torch.arange(leg_count) % corners.shape[0]]
        leg_noise = torch.randn(leg_count, 2, generator=generator) * 0.03
        leg_xy = chosen + leg_noise
        leg_z = -torch.rand(leg_count, 1, generator=generator) * 1.0
        points[seat_count + back_count :] = torch.cat([leg_xy, leg_z], dim=1)

        angle = (index % 16) * (2.0 * math.pi / 16.0)
        rot = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points = points @ rot.T
        points = points - points.mean(dim=0, keepdim=True)
        points = points / points.flatten().std().clamp_min(1e-6)
        return points


class PrimitivePointCloudDataset(Dataset[torch.Tensor]):
    """Finite normalized spheres and ellipsoids in [-1, 1]^3."""

    _RADII = (
        ("sphere", (0.35, 0.35, 0.35)),
        ("sphere", (0.55, 0.55, 0.55)),
        ("sphere", (0.75, 0.75, 0.75)),
        ("ellipsoid", (0.75, 0.45, 0.30)),
        ("ellipsoid", (0.35, 0.70, 0.50)),
        ("ellipsoid", (0.55, 0.30, 0.75)),
    )

    def __init__(self, num_shapes: int = 128, num_points: int = 64, seed: int = 0) -> None:
        if num_shapes <= 0:
            raise ValueError("num_shapes must be positive")
        if num_points < 8:
            raise ValueError("num_points must be at least 8")
        self.num_shapes = num_shapes
        self.num_points = num_points
        self.seed = seed

    def __len__(self) -> int:
        return self.num_shapes

    def primitive_name(self, index: int) -> str:
        if index < 0 or index >= self.num_shapes:
            raise IndexError(index)
        return self._RADII[index % len(self._RADII)][0]

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.num_shapes:
            raise IndexError(index)
        _, radii_tuple = self._RADII[index % len(self._RADII)]
        radii = torch.tensor(radii_tuple, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.seed + index)

        directions = torch.randn(self.num_points, 3, generator=generator)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        points = directions * radii.view(1, 3)

        angle = (index % 12) * (2.0 * math.pi / 12.0)
        rot = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points = points @ rot.T
        return points.clamp(-1.0, 1.0)


class StructuralObjectPointCloudDataset(Dataset[torch.Tensor]):
    """Synthetic chair/airplane/table point clouds for local structural smoke tests."""

    _OBJECTS = ("chair", "airplane", "table")

    def __init__(self, num_shapes: int = 128, num_points: int = 128, seed: int = 0) -> None:
        if num_shapes <= 0:
            raise ValueError("num_shapes must be positive")
        if num_points < 24:
            raise ValueError("num_points must be at least 24")
        self.num_shapes = num_shapes
        self.num_points = num_points
        self.seed = seed

    def __len__(self) -> int:
        return self.num_shapes

    def object_name(self, index: int) -> str:
        if index < 0 or index >= self.num_shapes:
            raise IndexError(index)
        return self._OBJECTS[index % len(self._OBJECTS)]

    @staticmethod
    def _allocate_counts(total: int, weights: tuple[float, ...]) -> list[int]:
        raw = torch.tensor(weights, dtype=torch.float32)
        counts = torch.floor(raw / raw.sum() * total).long()
        counts = counts.clamp_min(1)
        while int(counts.sum()) > total:
            largest = int(torch.argmax(counts).item())
            counts[largest] -= 1
        cursor = 0
        while int(counts.sum()) < total:
            counts[cursor % len(counts)] += 1
            cursor += 1
        return [int(value) for value in counts]

    @staticmethod
    def _sample_box(
        *,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        center_t = torch.tensor(center, dtype=torch.float32)
        half = torch.tensor(size, dtype=torch.float32) / 2.0
        points = (torch.rand(count, 3, generator=generator) * 2.0 - 1.0) * half
        axes = torch.randint(0, 3, size=(count,), generator=generator)
        signs = torch.randint(0, 2, size=(count,), generator=generator, dtype=torch.float32) * 2.0 - 1.0
        points[torch.arange(count), axes] = signs * half[axes]
        return points + center_t

    def _sample_from_boxes(
        self,
        boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        counts = self._allocate_counts(self.num_points, tuple(weight for _, _, weight in boxes))
        parts = [
            self._sample_box(center=center, size=size, count=count, generator=generator)
            for (center, size, _), count in zip(boxes, counts, strict=True)
        ]
        return torch.cat(parts, dim=0)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.num_shapes:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        name = self.object_name(index)

        if name == "chair":
            boxes = (
                ((0.0, 0.0, 0.0), (1.2, 1.0, 0.12), 5.0),
                ((0.0, 0.45, 0.65), (1.2, 0.12, 1.2), 5.0),
                ((-0.45, -0.35, -0.45), (0.14, 0.14, 0.9), 1.0),
                ((0.45, -0.35, -0.45), (0.14, 0.14, 0.9), 1.0),
                ((-0.45, 0.35, -0.45), (0.14, 0.14, 0.9), 1.0),
                ((0.45, 0.35, -0.45), (0.14, 0.14, 0.9), 1.0),
            )
        elif name == "airplane":
            boxes = (
                ((0.0, 0.0, 0.0), (1.8, 0.18, 0.18), 5.0),
                ((0.0, 0.0, 0.0), (0.65, 1.8, 0.08), 5.0),
                ((-0.75, 0.0, 0.28), (0.28, 0.55, 0.08), 1.5),
                ((-0.85, 0.0, 0.0), (0.18, 0.12, 0.65), 1.5),
                ((0.85, 0.0, 0.0), (0.22, 0.16, 0.16), 1.0),
            )
        elif name == "table":
            boxes = (
                ((0.0, 0.0, 0.45), (1.4, 1.1, 0.12), 7.0),
                ((-0.55, -0.4, -0.15), (0.14, 0.14, 1.2), 1.0),
                ((0.55, -0.4, -0.15), (0.14, 0.14, 1.2), 1.0),
                ((-0.55, 0.4, -0.15), (0.14, 0.14, 1.2), 1.0),
                ((0.55, 0.4, -0.15), (0.14, 0.14, 1.2), 1.0),
            )
        else:
            raise ValueError(f"unknown structural object: {name!r}")

        points = self._sample_from_boxes(boxes, generator=generator)
        jitter = torch.randn(points.shape, generator=generator) * 0.01
        points = points + jitter

        angle = (index % 18) * (2.0 * math.pi / 18.0)
        rot = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        points = points @ rot.T
        points = points - points.mean(dim=0, keepdim=True)
        points = points / points.abs().max().clamp_min(1e-6)
        return points.clamp(-1.0, 1.0)


class ShapeNetPointCloudDataset(Dataset[torch.Tensor]):
    """Load one category from ShapeNetCore.v2.PC15k-style .npy point clouds.

    Expected layout:
        root_dir/<synset_id>/<split>/*.npy

    Each .npy file should contain an array with shape (points, 3).
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        category: str = "chair",
        split: str = "train",
        num_points: int = 2048,
        seed: int = 0,
        normalize_per_shape: bool = True,
    ) -> None:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        synset_id = SHAPENET_CATEGORY_TO_SYNSET.get(category, category)
        category_dir = Path(root_dir) / synset_id / split
        if not category_dir.is_dir():
            raise FileNotFoundError(f"ShapeNet category split not found: {category_dir}")
        files = sorted(category_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no .npy point clouds found under: {category_dir}")
        self.files = files
        self.category = category
        self.synset_id = synset_id
        self.split = split
        self.num_points = num_points
        self.seed = seed
        self.normalize_per_shape = normalize_per_shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self.files):
            raise IndexError(index)
        array = np.load(self.files[index])
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"expected point cloud with shape (points, 3), got {array.shape}")
        points = torch.from_numpy(array.astype("float32", copy=False))
        if points.shape[0] < self.num_points:
            raise ValueError(f"{self.files[index]} has {points.shape[0]} points, need {self.num_points}")

        generator = torch.Generator().manual_seed(self.seed + index)
        chosen = torch.randperm(points.shape[0], generator=generator)[: self.num_points]
        points = points[chosen]
        if self.normalize_per_shape:
            points = points - points.mean(dim=0, keepdim=True)
            points = points / points.abs().max().clamp_min(1e-6)
        return points.clamp(-1.0, 1.0)
