import unittest

import torch

from fgc3d.targets import (
    diffusion_coefficients,
    make_noisy_points,
    prediction_to_target,
    prediction_to_v,
    prediction_to_x0_eps,
    training_target,
)


class PredictionTargetTest(unittest.TestCase):
    def test_prediction_targets_round_trip_to_x0_and_epsilon(self) -> None:
        x0 = torch.tensor(
            [
                [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
                [[-0.7, 0.8, 0.9], [0.2, -0.3, 0.4]],
            ],
            dtype=torch.float32,
        )
        epsilon = torch.tensor(
            [
                [[-0.3, 0.2, 0.1], [0.6, -0.5, 0.4]],
                [[0.9, 0.1, -0.8], [-0.4, 0.7, 0.2]],
            ],
            dtype=torch.float32,
        )
        t = torch.tensor([0.25, 0.75], dtype=torch.float32)
        alpha, sigma = diffusion_coefficients(t)
        x_t = make_noisy_points(x0, epsilon, alpha, sigma)

        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                predicted = training_target(target, x0, epsilon, alpha, sigma)
                recovered_x0, recovered_epsilon = prediction_to_x0_eps(
                    target=target,
                    prediction=predicted,
                    x_t=x_t,
                    alpha=alpha,
                    sigma=sigma,
                )

                self.assertTrue(torch.allclose(recovered_x0, x0, atol=1e-6))
                self.assertTrue(torch.allclose(recovered_epsilon, epsilon, atol=1e-6))

    def test_prediction_targets_round_trip_to_velocity(self) -> None:
        x0 = torch.randn(2, 5)
        epsilon = torch.randn(2, 5)
        t = torch.tensor([0.35, 0.65], dtype=torch.float32)
        alpha, sigma = diffusion_coefficients(t)
        x_t = make_noisy_points(x0, epsilon, alpha, sigma)
        expected_v = training_target("v", x0, epsilon, alpha, sigma)

        for target in ("x0", "epsilon", "v"):
            with self.subTest(target=target):
                predicted = training_target(target, x0, epsilon, alpha, sigma)
                recovered_v = prediction_to_v(
                    target=target,
                    prediction=predicted,
                    x_t=x_t,
                    alpha=alpha,
                    sigma=sigma,
                )
                self.assertTrue(torch.allclose(recovered_v, expected_v, atol=1e-6))

    def test_prediction_can_convert_to_each_loss_target(self) -> None:
        x0 = torch.randn(2, 4, 3)
        epsilon = torch.randn(2, 4, 3)
        t = torch.tensor([0.3, 0.8], dtype=torch.float32)
        alpha, sigma = diffusion_coefficients(t)
        x_t = make_noisy_points(x0, epsilon, alpha, sigma)

        for prediction_target in ("x0", "epsilon", "v"):
            prediction = training_target(prediction_target, x0, epsilon, alpha, sigma)
            for loss_target in ("x0", "epsilon", "v"):
                with self.subTest(prediction_target=prediction_target, loss_target=loss_target):
                    converted = prediction_to_target(
                        target=prediction_target,
                        prediction=prediction,
                        loss_target=loss_target,
                        x_t=x_t,
                        alpha=alpha,
                        sigma=sigma,
                    )
                    expected = training_target(loss_target, x0, epsilon, alpha, sigma)
                    self.assertTrue(torch.allclose(converted, expected, atol=1e-6))

    def test_unknown_prediction_target_is_rejected(self) -> None:
        x0 = torch.zeros(1, 2, 3)
        epsilon = torch.ones(1, 2, 3)
        t = torch.tensor([0.5])
        alpha, sigma = diffusion_coefficients(t)

        with self.assertRaisesRegex(ValueError, "prediction target"):
            training_target("score", x0, epsilon, alpha, sigma)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
