import unittest

import torch

from fgc3d.device import resolve_device
from fgc3d.train import train_toy_model


class DeviceTest(unittest.TestCase):
    def test_auto_prefers_cuda_only_when_available(self) -> None:
        device = resolve_device("auto")

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device.type, expected)

    def test_training_run_records_resolved_device(self) -> None:
        run = train_toy_model(
            prediction_target="x0",
            steps=1,
            batch_size=2,
            num_shapes=4,
            num_points=8,
            hidden_dim=16,
            device="cpu",
        )

        self.assertEqual(run.device.type, "cpu")
        self.assertEqual(next(run.model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
