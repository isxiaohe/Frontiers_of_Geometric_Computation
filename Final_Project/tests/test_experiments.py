import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

import torch

from fgc3d.experiments import run_primitives_generation_experiment


class PrimitivesGenerationExperimentTest(unittest.TestCase):
    def test_experiment_writes_samples_models_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            summary = run_primitives_generation_experiment(
                output_dir=output_dir,
                targets=("x0", "epsilon"),
                steps=1,
                batch_size=4,
                num_shapes=8,
                num_points=12,
                hidden_dim=16,
                num_samples=2,
                sample_steps=2,
                seed=101,
                write_plots=False,
            )

            self.assertEqual(set(summary["targets"].keys()), {"x0", "epsilon"})
            for target in ("x0", "epsilon"):
                self.assertTrue((output_dir / f"{target}_model.pt").exists())
                sample_path = output_dir / f"{target}_samples.pt"
                self.assertTrue(sample_path.exists())
                samples = torch.load(sample_path)
                self.assertEqual(samples.shape, (2, 12, 3))
                self.assertTrue(torch.isfinite(samples).all())
            self.assertTrue((output_dir / "summary.json").exists())

    def test_experiment_can_use_ddpm_schedule_without_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            summary = run_primitives_generation_experiment(
                output_dir=output_dir,
                targets=("epsilon",),
                steps=2,
                batch_size=2,
                num_shapes=4,
                num_points=12,
                hidden_dim=24,
                num_samples=2,
                sample_steps=4,
                seed=7,
                objective="diffusion",
                diffusion_schedule="ddpm",
                ddpm_steps=8,
                write_plots=False,
            )

            self.assertEqual(summary["objective"], "diffusion")
            self.assertEqual(summary["diffusion_schedule"], "ddpm")
            self.assertEqual(summary["ddpm_steps"], 8)
            self.assertTrue((output_dir / "epsilon_samples.pt").exists())

    def test_experiment_can_use_flow_objective_without_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            summary = run_primitives_generation_experiment(
                output_dir=output_dir,
                targets=("x0", "epsilon", "v"),
                objective="flow",
                steps=2,
                batch_size=2,
                num_shapes=4,
                num_points=12,
                hidden_dim=24,
                num_samples=2,
                sample_steps=4,
                seed=19,
                write_plots=False,
            )

            self.assertEqual(summary["objective"], "flow")
            self.assertEqual(set(summary["targets"].keys()), {"x0", "epsilon", "v"})
            for target in ("x0", "epsilon", "v"):
                self.assertTrue((output_dir / f"{target}_samples.pt").exists())

    def test_experiment_uses_same_train_and_sample_seed_for_each_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_primitives_generation_experiment(
                output_dir=Path(tmpdir),
                targets=("x0", "epsilon", "v"),
                steps=1,
                batch_size=2,
                num_shapes=4,
                num_points=12,
                hidden_dim=16,
                num_samples=2,
                sample_steps=2,
                seed=123,
                write_plots=False,
            )

            per_target = summary["targets"]
            train_seeds = {per_target[target]["train_seed"] for target in ("x0", "epsilon", "v")}
            sample_seeds = {per_target[target]["sample_seed"] for target in ("x0", "epsilon", "v")}

            self.assertEqual(train_seeds, {123})
            self.assertEqual(sample_seeds, {1123})

    def test_experiment_records_chamfer_metrics_for_each_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_primitives_generation_experiment(
                output_dir=Path(tmpdir),
                targets=("x0",),
                steps=1,
                batch_size=2,
                num_shapes=4,
                num_points=12,
                hidden_dim=16,
                num_samples=2,
                sample_steps=2,
                seed=125,
                write_plots=False,
            )

            metrics = summary["targets"]["x0"]["metrics"]

            self.assertIn("mmd_cd", metrics)
            self.assertIn("coverage", metrics)
            self.assertGreaterEqual(metrics["mmd_cd"], 0.0)
            self.assertGreaterEqual(metrics["coverage"], 0.0)
            self.assertLessEqual(metrics["coverage"], 1.0)

    def test_train_generate_primitives_cli_accepts_target_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_generate_primitives.py",
                    "--output-dir",
                    str(output_dir),
                    "--targets",
                    "x0",
                    "--steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--num-shapes",
                    "4",
                    "--num-points",
                    "12",
                    "--hidden-dim",
                    "16",
                    "--num-samples",
                    "2",
                    "--sample-steps",
                    "2",
                    "--no-plots",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"x0"', result.stdout)
            self.assertTrue((output_dir / "x0_samples.pt").exists())
            self.assertFalse((output_dir / "epsilon_samples.pt").exists())

    def test_train_generate_primitives_cli_accepts_flow_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_generate_primitives.py",
                    "--output-dir",
                    str(output_dir),
                    "--objective",
                    "flow",
                    "--targets",
                    "v",
                    "--steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--num-shapes",
                    "4",
                    "--num-points",
                    "12",
                    "--hidden-dim",
                    "16",
                    "--num-samples",
                    "2",
                    "--sample-steps",
                    "2",
                    "--no-plots",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"objective": "flow"', result.stdout)
            self.assertTrue((output_dir / "v_samples.pt").exists())

    def test_train_generate_shapenet_cli_writes_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            train_dir = data_root / "03797390" / "train"
            train_dir.mkdir(parents=True)
            for index in range(4):
                points = torch.randn(32, 3).numpy().astype("float32")
                (train_dir / f"mug_{index}.npy").parent.mkdir(parents=True, exist_ok=True)
                import numpy as np

                np.save(train_dir / f"mug_{index}.npy", points)
            output_dir = Path(tmpdir) / "out"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_generate_shapenet_category.py",
                    "--output-dir",
                    str(output_dir),
                    "--data-root",
                    str(data_root),
                    "--category",
                    "mug",
                    "--prediction-target",
                    "x0",
                    "--steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--num-points",
                    "12",
                    "--hidden-dim",
                    "16",
                    "--num-samples",
                    "2",
                    "--sample-steps",
                    "2",
                    "--no-plots",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output_dir / "x0_samples.pt").exists())
            self.assertTrue((output_dir / "summary.json").exists())

    def test_train_generate_shapenet_cli_defaults_to_chair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            train_dir = data_root / "03001627" / "train"
            train_dir.mkdir(parents=True)
            import numpy as np

            for index in range(4):
                points = torch.randn(32, 3).numpy().astype("float32")
                np.save(train_dir / f"chair_{index}.npy", points)
            output_dir = Path(tmpdir) / "out"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_generate_shapenet_category.py",
                    "--output-dir",
                    str(output_dir),
                    "--data-root",
                    str(data_root),
                    "--prediction-target",
                    "x0",
                    "--steps",
                    "1",
                    "--batch-size",
                    "2",
                    "--num-points",
                    "12",
                    "--hidden-dim",
                    "16",
                    "--num-samples",
                    "2",
                    "--sample-steps",
                    "2",
                    "--no-plots",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"category": "chair"', result.stdout)
            self.assertIn('"objective": "flow"', result.stdout)
            self.assertIn('"sampling_condition_number"', result.stdout)
            self.assertTrue((output_dir / "x0_samples.pt").exists())


if __name__ == "__main__":
    unittest.main()
