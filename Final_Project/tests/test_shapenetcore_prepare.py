import tempfile
import unittest
from pathlib import Path

import numpy as np

from fgc3d.shapenetcore import convert_category_to_pc15k, load_obj_mesh, sample_mesh_points


class ShapeNetCorePrepareTest(unittest.TestCase):
    def test_load_obj_mesh_parses_vertices_and_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "model_normalized.obj"
            obj_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    ]
                ),
                encoding="utf-8",
            )

            vertices, faces = load_obj_mesh(obj_path)

            self.assertEqual(vertices.shape, (3, 3))
            self.assertEqual(faces.shape, (1, 3))

    def test_sample_mesh_points_returns_finite_point_cloud(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        points = sample_mesh_points(vertices, faces, num_points=16, seed=5)

        self.assertEqual(points.shape, (16, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertLessEqual(float(np.abs(points).max()), 1.0)

    def test_convert_category_to_pc15k_writes_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw"
            model_dir = raw_root / "03001627" / "model_a" / "models"
            model_dir.mkdir(parents=True)
            (model_dir / "model_normalized.obj").write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "v 0 0 1",
                        "f 1 2 3",
                        "f 1 2 4",
                    ]
                ),
                encoding="utf-8",
            )
            output_root = Path(tmpdir) / "pc15k"

            written = convert_category_to_pc15k(
                raw_root=raw_root,
                output_root=output_root,
                category="chair",
                split="train",
                num_points=32,
                max_models=1,
                seed=11,
            )

            self.assertEqual(len(written), 1)
            expected_path = output_root / "03001627" / "train" / "model_a.npy"
            self.assertEqual(written[0], expected_path)
            points = np.load(expected_path)
            self.assertEqual(points.shape, (32, 3))


if __name__ == "__main__":
    unittest.main()
