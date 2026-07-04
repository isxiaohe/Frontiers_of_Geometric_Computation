import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from fgc3d.data import PrimitivePointCloudDataset, ShapeNetPointCloudDataset, StructuralObjectPointCloudDataset


class PrimitivePointCloudDatasetTest(unittest.TestCase):
    def test_primitives_are_normalized_point_clouds(self) -> None:
        dataset = PrimitivePointCloudDataset(num_shapes=12, num_points=64, seed=23)
        point_cloud = dataset[0]

        self.assertEqual(point_cloud.shape, (64, 3))
        self.assertLessEqual(float(point_cloud.abs().max()), 1.0)
        self.assertTrue(torch.isfinite(point_cloud).all())

    def test_primitives_include_spheres_and_ellipsoids(self) -> None:
        dataset = PrimitivePointCloudDataset(num_shapes=12, num_points=64, seed=23)
        kinds = {dataset.primitive_name(index) for index in range(len(dataset))}

        self.assertIn("sphere", kinds)
        self.assertIn("ellipsoid", kinds)


class StructuralObjectPointCloudDatasetTest(unittest.TestCase):
    def test_structural_objects_are_normalized_point_clouds(self) -> None:
        dataset = StructuralObjectPointCloudDataset(num_shapes=9, num_points=96, seed=47)
        point_cloud = dataset[0]

        self.assertEqual(point_cloud.shape, (96, 3))
        self.assertLessEqual(float(point_cloud.abs().max()), 1.0)
        self.assertTrue(torch.isfinite(point_cloud).all())

    def test_structural_objects_include_chair_airplane_and_table(self) -> None:
        dataset = StructuralObjectPointCloudDataset(num_shapes=9, num_points=96, seed=47)
        kinds = {dataset.object_name(index) for index in range(len(dataset))}

        self.assertEqual(kinds, {"chair", "airplane", "table"})


class ShapeNetPointCloudDatasetTest(unittest.TestCase):
    def test_shapenet_pc15k_layout_loads_one_category(self) -> None:
        with TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "03001627" / "train"
            category_dir.mkdir(parents=True)
            points = np.linspace(-1.0, 1.0, num=90, dtype=np.float32).reshape(30, 3)
            np.save(category_dir / "chair_000.npy", points)

            dataset = ShapeNetPointCloudDataset(
                root_dir=Path(tmpdir),
                category="chair",
                split="train",
                num_points=12,
                seed=31,
            )

            point_cloud = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(point_cloud.shape, (12, 3))
            self.assertLessEqual(float(point_cloud.abs().max()), 1.0)
            self.assertTrue(torch.isfinite(point_cloud).all())


if __name__ == "__main__":
    unittest.main()
