import unittest

from fgc3d.overfit import run_fixed_batch_overfit


class FixedBatchOverfitTest(unittest.TestCase):
    def test_fixed_batch_overfit_reduces_loss(self) -> None:
        summary = run_fixed_batch_overfit(
            prediction_target="x0",
            objective="flow",
            loss_mode="target",
            model_name="tiny",
            steps=30,
            batch_size=2,
            num_points=16,
            hidden_dim=48,
            lr=5e-3,
            seed=41,
        )

        self.assertLess(summary["final_loss"], summary["first_loss"])
        self.assertGreater(summary["loss_reduction"], 0.0)
        self.assertEqual(summary["objective"], "flow")

    def test_fixed_batch_overfit_can_use_diffusion_reference(self) -> None:
        summary = run_fixed_batch_overfit(
            prediction_target="epsilon",
            objective="diffusion",
            loss_mode="target",
            model_name="tiny",
            steps=5,
            batch_size=2,
            num_points=16,
            hidden_dim=32,
            lr=1e-3,
            seed=43,
        )

        self.assertEqual(summary["objective"], "diffusion")
        self.assertGreater(summary["first_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
