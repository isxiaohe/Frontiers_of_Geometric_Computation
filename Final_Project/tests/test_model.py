import unittest

import torch

from fgc3d.model import MiniVoxelPointDenoiser, PVDLiteDenoiser, PointNetContextDenoiser, TinyPointDenoiser


class TinyPointDenoiserTest(unittest.TestCase):
    def test_tiny_point_denoiser_preserves_point_cloud_shape(self) -> None:
        model = TinyPointDenoiser(hidden_dim=32, time_features=16)
        x_t = torch.randn(4, 12, 3)
        t = torch.linspace(0.1, 0.9, steps=4)

        output = model(x_t, t)

        self.assertEqual(output.shape, x_t.shape)
        self.assertTrue(torch.isfinite(output).all())


class PointNetContextDenoiserTest(unittest.TestCase):
    def test_context_denoiser_preserves_point_cloud_shape(self) -> None:
        model = PointNetContextDenoiser(hidden_dim=32, context_dim=48, time_features=16)
        x_t = torch.randn(4, 12, 3)
        t = torch.linspace(0.1, 0.9, steps=4)

        output = model(x_t, t)

        self.assertEqual(output.shape, x_t.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_context_denoiser_is_permutation_equivariant(self) -> None:
        torch.manual_seed(5)
        model = PointNetContextDenoiser(hidden_dim=32, context_dim=48, time_features=16)
        model.eval()
        x_t = torch.randn(2, 10, 3)
        t = torch.tensor([0.2, 0.7])
        permutation = torch.tensor([3, 0, 9, 1, 5, 2, 8, 4, 7, 6])

        output = model(x_t, t)
        permuted_output = model(x_t[:, permutation], t)

        self.assertTrue(torch.allclose(permuted_output, output[:, permutation], atol=1e-6))


class MiniVoxelPointDenoiserTest(unittest.TestCase):
    def test_voxel_point_denoiser_preserves_point_cloud_shape(self) -> None:
        model = MiniVoxelPointDenoiser(hidden_dim=24, time_features=16, voxel_resolution=6)
        x_t = torch.randn(3, 14, 3)
        t = torch.linspace(0.1, 0.9, steps=3)

        output = model(x_t, t)

        self.assertEqual(output.shape, x_t.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_voxel_point_denoiser_is_permutation_equivariant(self) -> None:
        torch.manual_seed(7)
        model = MiniVoxelPointDenoiser(hidden_dim=24, time_features=16, voxel_resolution=6)
        model.eval()
        x_t = torch.randn(2, 12, 3)
        t = torch.tensor([0.2, 0.7])
        permutation = torch.tensor([4, 0, 11, 2, 8, 1, 10, 3, 9, 6, 5, 7])

        output = model(x_t, t)
        permuted_output = model(x_t[:, permutation], t)

        self.assertTrue(torch.allclose(permuted_output, output[:, permutation], atol=1e-5))

    def test_voxelization_is_translation_invariant_like_pvd(self) -> None:
        model = MiniVoxelPointDenoiser(hidden_dim=16, time_features=16, voxel_resolution=6)
        features = torch.randn(1, 8, 16)
        coords = torch.randn(1, 8, 3) * 0.25
        shift = torch.tensor([[[3.0, -2.0, 1.0]]])

        grid, normalized_coords = model._voxelize(features, coords)
        shifted_grid, shifted_normalized_coords = model._voxelize(features, coords + shift)

        self.assertTrue(torch.allclose(shifted_grid, grid, atol=1e-6))
        self.assertTrue(torch.allclose(shifted_normalized_coords, normalized_coords, atol=1e-6))


class PVDLiteDenoiserTest(unittest.TestCase):
    def test_pvdlite_denoiser_preserves_point_cloud_shape(self) -> None:
        model = PVDLiteDenoiser(hidden_dim=24, time_features=16, voxel_resolution=6)
        x_t = torch.randn(3, 14, 3)
        t = torch.linspace(0.1, 0.9, steps=3)

        output = model(x_t, t)

        self.assertEqual(output.shape, x_t.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_pvdlite_denoiser_is_permutation_equivariant(self) -> None:
        torch.manual_seed(11)
        model = PVDLiteDenoiser(hidden_dim=24, time_features=16, voxel_resolution=6)
        model.eval()
        x_t = torch.randn(2, 12, 3)
        t = torch.tensor([0.2, 0.7])
        permutation = torch.tensor([4, 0, 11, 2, 8, 1, 10, 3, 9, 6, 5, 7])

        output = model(x_t, t)
        permuted_output = model(x_t[:, permutation], t)

        self.assertTrue(torch.allclose(permuted_output, output[:, permutation], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
