import unittest

import torch

from fgc3d.manifold import ProjectedSphereDataset, VectorDenoiser, run_projected_sphere_training
from fgc3d.train import train_step


class ProjectedSphereTest(unittest.TestCase):
    def test_projection_has_orthonormal_columns(self) -> None:
        dataset = ProjectedSphereDataset(num_samples=8, intrinsic_dim=2, ambient_dim=16, seed=3)
        gram = dataset.projection.T @ dataset.projection

        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-6))

    def test_samples_live_on_projected_unit_sphere(self) -> None:
        dataset = ProjectedSphereDataset(num_samples=8, intrinsic_dim=2, ambient_dim=16, seed=3)
        sample = dataset[0]

        self.assertEqual(sample.shape, (16,))
        self.assertAlmostEqual(float(sample.norm()), 1.0, places=5)

    def test_vector_training_step_updates_for_each_prediction_target(self) -> None:
        dataset = ProjectedSphereDataset(num_samples=6, intrinsic_dim=2, ambient_dim=16, seed=5)
        batch = torch.stack([dataset[i] for i in range(6)])

        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                torch.manual_seed(11)
                model = VectorDenoiser(ambient_dim=16, hidden_dim=64, time_features=16)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = [param.detach().clone() for param in model.parameters()]

                metrics = train_step(model, optimizer, batch, prediction_target=target, loss_mode="v")

                self.assertGreater(metrics.loss, 0)
                self.assertTrue(
                    any(
                        not torch.allclose(old, new.detach())
                        for old, new in zip(before, model.parameters(), strict=True)
                    )
                )

    def test_projected_sphere_training_runs(self) -> None:
        metrics = run_projected_sphere_training(
            prediction_target="x0",
            steps=3,
            batch_size=8,
            num_samples=32,
            intrinsic_dim=2,
            ambient_dim=16,
            hidden_dim=64,
            seed=13,
            loss_mode="v",
        )

        self.assertEqual(metrics.prediction_target, "x0")
        self.assertEqual(len(metrics.losses), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses))


if __name__ == "__main__":
    unittest.main()
