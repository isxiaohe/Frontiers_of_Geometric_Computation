import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fgc3d.data import SyntheticPointCloudDataset
from fgc3d.model import TinyPointDenoiser
from fgc3d.sample import sample_points
from fgc3d.scheduler import DDPMDiffusionScheduler
from fgc3d.train import run_toy_training, train_step


class TrainingTest(unittest.TestCase):
    def test_train_step_updates_parameters_for_each_prediction_target(self) -> None:
        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                torch.manual_seed(7)
                dataset = SyntheticPointCloudDataset(num_shapes=4, num_points=16, seed=11)
                batch = torch.stack([dataset[i] for i in range(4)])
                model = TinyPointDenoiser(hidden_dim=32, time_features=16)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = [param.detach().clone() for param in model.parameters()]

                metrics = train_step(model, optimizer, batch, prediction_target=target)

                self.assertGreater(metrics.loss, 0)
                self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
                self.assertTrue(
                    any(
                        not torch.allclose(old, new.detach())
                        for old, new in zip(before, model.parameters(), strict=True)
                    )
                )

    def test_train_step_runs_each_loss_mode(self) -> None:
        for prediction_target in ("x0", "epsilon", "v"):
            for loss_mode in ("target", "x0", "epsilon", "v"):
                with self.subTest(prediction_target=prediction_target, loss_mode=loss_mode):
                    torch.manual_seed(9)
                    dataset = SyntheticPointCloudDataset(num_shapes=4, num_points=16, seed=11)
                    batch = torch.stack([dataset[i] for i in range(4)])
                    model = TinyPointDenoiser(hidden_dim=32, time_features=16)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    metrics = train_step(
                        model,
                        optimizer,
                        batch,
                        prediction_target=prediction_target,
                        loss_mode=loss_mode,
                    )

                    self.assertGreater(metrics.loss, 0)
                    self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))

    def test_toy_training_loop_runs_for_each_prediction_target(self) -> None:
        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                metrics = run_toy_training(
                    prediction_target=target,
                    steps=3,
                    batch_size=4,
                    num_shapes=12,
                    num_points=16,
                    hidden_dim=32,
                    seed=13,
                )

                self.assertEqual(len(metrics.losses), 3)
                self.assertTrue(all(loss > 0 for loss in metrics.losses))
                self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in metrics.losses))

    def test_train_toy_cli_runs_for_each_prediction_target(self) -> None:
        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fgc3d.cli.train_toy",
                        "--prediction-target",
                        target,
                        "--steps",
                        "2",
                        "--batch-size",
                        "4",
                        "--num-shapes",
                        "8",
                        "--num-points",
                        "12",
                        "--hidden-dim",
                        "32",
                        "--seed",
                        "17",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                self.assertIn(f'"prediction_target": "{target}"', result.stdout)
                self.assertIn('"objective": "flow"', result.stdout)
                self.assertIn('"losses":', result.stdout)

    def test_train_toy_cli_runs_each_unified_loss_mode(self) -> None:
        for loss_mode in ("x0", "epsilon", "v"):
            with self.subTest(loss_mode=loss_mode):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fgc3d.cli.train_toy",
                        "--prediction-target",
                        "epsilon",
                        "--loss-mode",
                        loss_mode,
                        "--steps",
                        "2",
                        "--batch-size",
                        "4",
                        "--num-shapes",
                        "8",
                        "--num-points",
                        "12",
                        "--hidden-dim",
                        "32",
                        "--seed",
                        "17",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                self.assertIn(f'"loss_mode": "{loss_mode}"', result.stdout)

    def test_train_toy_cli_runs_ddpm_diffusion_schedule(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--prediction-target",
                "epsilon",
                "--objective",
                "diffusion",
                "--diffusion-schedule",
                "ddpm",
                "--ddpm-steps",
                "16",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "8",
                "--num-points",
                "12",
                "--hidden-dim",
                "32",
                "--seed",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"diffusion_schedule": "ddpm"', result.stdout)

    def test_train_toy_cli_runs_flow_matching_on_primitives(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--objective",
                "flow",
                "--dataset",
                "primitives",
                "--prediction-target",
                "v",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "8",
                "--num-points",
                "12",
                "--hidden-dim",
                "32",
                "--seed",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"prediction_target": "v"', result.stdout)
        self.assertIn('"objective": "flow"', result.stdout)

    def test_train_toy_cli_runs_structural_objects(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--objective",
                "flow",
                "--dataset",
                "structural",
                "--model",
                "pvdlite",
                "--voxel-resolution",
                "6",
                "--prediction-target",
                "x0",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "9",
                "--num-points",
                "24",
                "--hidden-dim",
                "24",
                "--seed",
                "29",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"dataset": "structural"', result.stdout)
        self.assertIn('"model": "pvdlite"', result.stdout)

    def test_train_toy_cli_runs_pointnet_context_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--model",
                "pointnet",
                "--dataset",
                "primitives",
                "--prediction-target",
                "x0",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "8",
                "--num-points",
                "12",
                "--hidden-dim",
                "32",
                "--seed",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"model": "pointnet"', result.stdout)

    def test_train_toy_cli_runs_voxel_point_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--model",
                "voxelpoint",
                "--voxel-resolution",
                "6",
                "--dataset",
                "primitives",
                "--prediction-target",
                "epsilon",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "8",
                "--num-points",
                "12",
                "--hidden-dim",
                "24",
                "--seed",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"model": "voxelpoint"', result.stdout)

    def test_train_toy_cli_runs_pvdlite_model(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fgc3d.cli.train_toy",
                "--model",
                "pvdlite",
                "--voxel-resolution",
                "6",
                "--dataset",
                "primitives",
                "--prediction-target",
                "x0",
                "--steps",
                "2",
                "--batch-size",
                "4",
                "--num-shapes",
                "8",
                "--num-points",
                "12",
                "--hidden-dim",
                "24",
                "--seed",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn('"model": "pvdlite"', result.stdout)

    def test_train_toy_cli_runs_shapenet_category(self) -> None:
        with TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "03001627" / "train"
            category_dir.mkdir(parents=True)
            for index in range(4):
                points = np.random.default_rng(index).normal(size=(40, 3)).astype("float32")
                np.save(category_dir / f"chair_{index:03d}.npy", points)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fgc3d.cli.train_toy",
                    "--dataset",
                    "shapenet",
                    "--data-root",
                    tmpdir,
                    "--category",
                    "chair",
                    "--prediction-target",
                    "x0",
                    "--steps",
                    "2",
                    "--batch-size",
                    "2",
                    "--num-points",
                    "12",
                    "--hidden-dim",
                    "24",
                    "--seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"dataset": "shapenet"', result.stdout)
            self.assertIn('"category": "chair"', result.stdout)

    def test_sampler_runs_for_each_prediction_target(self) -> None:
        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                model = TinyPointDenoiser(hidden_dim=32, time_features=16)
                points = sample_points(
                    model,
                    prediction_target=target,
                    num_shapes=3,
                    num_points=10,
                    steps=3,
                    seed=19,
                )

                self.assertEqual(points.shape, (3, 10, 3))
                self.assertTrue(torch.isfinite(points).all())

    def test_sampler_runs_with_ddpm_scheduler(self) -> None:
        model = TinyPointDenoiser(hidden_dim=32, time_features=16)
        scheduler = DDPMDiffusionScheduler(num_train_steps=8)

        points = sample_points(
            model,
            prediction_target="epsilon",
            num_shapes=3,
            num_points=10,
            steps=8,
            seed=19,
            scheduler=scheduler,
        )

        self.assertEqual(points.shape, (3, 10, 3))
        self.assertTrue(torch.isfinite(points).all())

    def test_ddpm_sampler_respects_requested_sample_steps(self) -> None:
        class CountingDenoiser(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0

            def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return torch.zeros_like(x_t) + self.bias

        model = CountingDenoiser()
        scheduler = DDPMDiffusionScheduler(num_train_steps=8)

        sample_points(
            model,
            prediction_target="epsilon",
            num_shapes=2,
            num_points=6,
            steps=3,
            seed=23,
            scheduler=scheduler,
        )

        self.assertEqual(model.calls, 3)


if __name__ == "__main__":
    unittest.main()
