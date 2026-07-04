import unittest

import torch

from fgc3d.data import PrimitivePointCloudDataset
from fgc3d.flow_matching import (
    EulerFlowScheduler,
    flow_matching_condition_number,
    flow_matching_path,
    flow_matching_prediction_to_target,
    flow_matching_target,
    flow_matching_training_target,
    sample_flow,
)
from fgc3d.model import TinyPointDenoiser
from fgc3d.train import train_flow_step, train_toy_model


class FlowMatchingTest(unittest.TestCase):
    def test_ot_path_matches_noise_and_data_endpoints(self) -> None:
        x0 = torch.randn(2, 5, 3)
        z = torch.randn(2, 5, 3)

        start = flow_matching_path(x0, z, torch.zeros(2))
        end = flow_matching_path(x0, z, torch.ones(2))

        self.assertTrue(torch.allclose(start, z))
        self.assertTrue(torch.allclose(end, x0))

    def test_ot_velocity_target_is_data_minus_noise(self) -> None:
        x0 = torch.randn(2, 5, 3)
        z = torch.randn(2, 5, 3)

        target = flow_matching_target(x0, z)

        self.assertTrue(torch.allclose(target, x0 - z))

    def test_flow_parameterizations_convert_to_velocity_target(self) -> None:
        x0 = torch.randn(2, 5, 3)
        z = torch.randn(2, 5, 3)
        t = torch.tensor([0.25, 0.75])
        x_t = flow_matching_path(x0, z, t)
        expected_velocity = flow_matching_training_target("v", x0, z, t)

        for prediction_target, prediction in (
            ("x0", x0),
            ("epsilon", z),
            ("v", x0 - z),
        ):
            with self.subTest(prediction_target=prediction_target):
                converted = flow_matching_prediction_to_target(
                    target=prediction_target,
                    prediction=prediction,
                    loss_target="v",
                    x_t=x_t,
                    t=t,
                )

                self.assertTrue(torch.allclose(converted, expected_velocity, atol=1e-6))

    def test_flow_step_updates_model_on_primitives(self) -> None:
        dataset = PrimitivePointCloudDataset(num_shapes=4, num_points=16, seed=31)
        batch = torch.stack([dataset[i] for i in range(4)])
        model = TinyPointDenoiser(hidden_dim=32, time_features=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [param.detach().clone() for param in model.parameters()]

        metrics = train_flow_step(model, optimizer, batch)

        self.assertGreater(metrics.loss, 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
        self.assertTrue(
            any(
                not torch.allclose(old, new.detach())
                for old, new in zip(before, model.parameters(), strict=True)
            )
        )

    def test_flow_step_runs_each_prediction_and_loss_mode(self) -> None:
        dataset = PrimitivePointCloudDataset(num_shapes=4, num_points=16, seed=31)
        batch = torch.stack([dataset[i] for i in range(4)])

        for prediction_target in ("x0", "epsilon", "v"):
            for loss_mode in ("target", "x0", "epsilon", "v"):
                with self.subTest(prediction_target=prediction_target, loss_mode=loss_mode):
                    model = TinyPointDenoiser(hidden_dim=32, time_features=16)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    metrics = train_flow_step(
                        model,
                        optimizer,
                        batch,
                        prediction_target=prediction_target,
                        loss_mode=loss_mode,
                    )

                    self.assertGreater(metrics.loss, 0)
                    self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))

    def test_euler_scheduler_samples_finite_point_clouds(self) -> None:
        model = TinyPointDenoiser(hidden_dim=32, time_features=16)
        scheduler = EulerFlowScheduler(num_steps=4)

        samples = sample_flow(
            model,
            scheduler=scheduler,
            prediction_target="v",
            num_shapes=3,
            num_points=12,
            seed=37,
        )

        self.assertEqual(samples.shape, (3, 12, 3))
        self.assertTrue(torch.isfinite(samples).all())

    def test_euler_scheduler_evaluates_interior_midpoints(self) -> None:
        scheduler = EulerFlowScheduler(num_steps=4)

        times = scheduler.times(device=torch.device("cpu"), dtype=torch.float32)

        self.assertTrue(torch.all(times > 0))
        self.assertTrue(torch.all(times < 1))
        self.assertTrue(torch.allclose(times, torch.tensor([0.125, 0.375, 0.625, 0.875])))

    def test_condition_number_reports_endpoint_parameterization_risk(self) -> None:
        scheduler = EulerFlowScheduler(num_steps=4)

        self.assertEqual(flow_matching_condition_number("v", scheduler), 1.0)
        self.assertEqual(flow_matching_condition_number("epsilon", scheduler), 8.0)
        self.assertEqual(flow_matching_condition_number("x0", scheduler), 8.0)

    def test_trained_flow_model_can_be_sampled(self) -> None:
        run = train_toy_model(
            prediction_target="v",
            objective="flow",
            dataset_name="primitives",
            steps=3,
            batch_size=4,
            num_shapes=12,
            num_points=16,
            hidden_dim=32,
            seed=41,
        )
        samples = sample_flow(
            run.model,
            scheduler=EulerFlowScheduler(num_steps=4),
            prediction_target="v",
            num_shapes=2,
            num_points=16,
            seed=43,
        )

        self.assertEqual(samples.shape, (2, 16, 3))
        self.assertTrue(torch.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
