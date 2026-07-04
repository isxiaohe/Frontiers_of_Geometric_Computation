import unittest

import torch

from fgc3d.metrics import chamfer_distance, point_cloud_distribution_metrics


class PointCloudMetricsTest(unittest.TestCase):
    def test_chamfer_distance_is_zero_for_identical_clouds(self) -> None:
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        distance = chamfer_distance(points, points)

        self.assertEqual(float(distance), 0.0)

    def test_distribution_metrics_are_finite_and_bounded(self) -> None:
        reference = torch.randn(3, 8, 3)
        samples = reference.clone()

        metrics = point_cloud_distribution_metrics(samples=samples, reference=reference)

        self.assertAlmostEqual(metrics["mmd_cd"], 0.0, places=6)
        self.assertGreaterEqual(metrics["coverage"], 0.0)
        self.assertLessEqual(metrics["coverage"], 1.0)

    def test_coverage_counts_unique_reference_clouds_matched_by_samples(self) -> None:
        samples = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
                [[10.1, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        metrics = point_cloud_distribution_metrics(samples=samples, reference=reference)

        self.assertAlmostEqual(metrics["coverage"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
