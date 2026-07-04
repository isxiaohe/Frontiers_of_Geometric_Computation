import unittest

import torch

from fgc3d.scheduler import DDPMDiffusionScheduler, VPDiffusionScheduler


class VPDiffusionSchedulerTest(unittest.TestCase):
    def test_scheduler_produces_reverse_time_pairs(self) -> None:
        scheduler = VPDiffusionScheduler(num_steps=4, t_min=0.02, t_max=0.98)

        pairs = list(scheduler.time_pairs(device=torch.device("cpu"), dtype=torch.float32))

        self.assertEqual(len(pairs), 4)
        self.assertAlmostEqual(float(pairs[0][0]), 0.98, places=6)
        self.assertAlmostEqual(float(pairs[-1][1]), 0.02, places=6)
        self.assertTrue(all(float(t) > float(next_t) for t, next_t in pairs))

    def test_scheduler_step_preserves_shape_and_finiteness(self) -> None:
        scheduler = VPDiffusionScheduler(num_steps=4)
        x_t = torch.randn(2, 8, 3)
        prediction = torch.randn(2, 8, 3)
        t = torch.full((2,), 0.8)
        next_t = torch.full((2,), 0.6)

        updated = scheduler.ddim_step(
            x_t=x_t,
            prediction=prediction,
            prediction_target="epsilon",
            t=t,
            next_t=next_t,
        )

        self.assertEqual(updated.shape, x_t.shape)
        self.assertTrue(torch.isfinite(updated).all())


class DDPMDiffusionSchedulerTest(unittest.TestCase):
    def test_ddpm_q_sample_matches_closed_form_endpoints(self) -> None:
        scheduler = DDPMDiffusionScheduler(num_train_steps=5, beta_start=0.1, beta_end=0.2)
        x0 = torch.randn(2, 6, 3)
        noise = torch.randn_like(x0)
        t_index = torch.tensor([0, 4], dtype=torch.long)

        x_t = scheduler.q_sample(x0=x0, t_index=t_index, noise=noise)
        alpha, sigma = scheduler.training_coefficients(t_index, x0.shape)
        expected = alpha * x0 + sigma * noise

        self.assertTrue(torch.allclose(x_t, expected))
        self.assertEqual(x_t.shape, x0.shape)

    def test_ddpm_p_sample_step_preserves_shape_and_finiteness(self) -> None:
        scheduler = DDPMDiffusionScheduler(num_train_steps=5, beta_start=0.1, beta_end=0.2)
        x_t = torch.randn(2, 6, 3)
        prediction = torch.randn_like(x_t)
        t_index = torch.tensor([4, 3], dtype=torch.long)
        noise = torch.randn_like(x_t)

        sample, pred_x0 = scheduler.p_sample_step(
            x_t=x_t,
            prediction=prediction,
            prediction_target="epsilon",
            t_index=t_index,
            noise=noise,
            clip_denoised=True,
        )

        self.assertEqual(sample.shape, x_t.shape)
        self.assertEqual(pred_x0.shape, x_t.shape)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertLessEqual(float(pred_x0.abs().max()), 1.0)

    def test_ddpm_reverse_timesteps_are_descending(self) -> None:
        scheduler = DDPMDiffusionScheduler(num_train_steps=5)

        timesteps = list(scheduler.reverse_timesteps(device=torch.device("cpu")))

        self.assertEqual([int(t) for t in timesteps], [4, 3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
