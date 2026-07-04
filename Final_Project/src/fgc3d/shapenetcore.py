"""Utilities for preparing ShapeNetCore meshes as point-cloud .npy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import SHAPENET_CATEGORY_TO_SYNSET


def category_to_synset(category: str) -> str:
    return SHAPENET_CATEGORY_TO_SYNSET.get(category, category)


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangular faces from a Wavefront OBJ file."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                indices = [_parse_obj_index(token, len(vertices)) for token in parts[1:]]
                for offset in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[offset], indices[offset + 1]])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/")[0])
    if raw > 0:
        return raw - 1
    return vertex_count + raw


def sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_points: int,
    seed: int,
) -> np.ndarray:
    """Sample a normalized point cloud from mesh triangles or vertices."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    rng = np.random.default_rng(seed)
    if faces.size == 0:
        chosen = rng.choice(vertices.shape[0], size=num_points, replace=True)
        points = vertices[chosen]
    else:
        triangles = vertices[faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
        if float(areas.sum()) <= 0.0:
            chosen = rng.choice(vertices.shape[0], size=num_points, replace=True)
            points = vertices[chosen]
        else:
            probabilities = areas / areas.sum()
            tri_indices = rng.choice(triangles.shape[0], size=num_points, replace=True, p=probabilities)
            selected = triangles[tri_indices]
            u = rng.random((num_points, 1), dtype=np.float32)
            v = rng.random((num_points, 1), dtype=np.float32)
            flip = (u + v) > 1.0
            u[flip] = 1.0 - u[flip]
            v[flip] = 1.0 - v[flip]
            points = selected[:, 0] + u * (selected[:, 1] - selected[:, 0]) + v * (selected[:, 2] - selected[:, 0])

    points = points.astype(np.float32, copy=False)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.abs(points).max()
    if scale > 0:
        points = points / scale
    return np.clip(points, -1.0, 1.0).astype(np.float32, copy=False)


def find_category_obj_files(raw_root: Path, category: str) -> list[Path]:
    synset = category_to_synset(category)
    category_root = raw_root / synset
    if not category_root.is_dir():
        raise FileNotFoundError(f"ShapeNetCore category directory not found: {category_root}")
    obj_files = sorted(category_root.glob("*/models/model_normalized.obj"))
    if not obj_files:
        obj_files = sorted(category_root.rglob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"no OBJ files found under: {category_root}")
    return obj_files


def convert_category_to_pc15k(
    *,
    raw_root: Path,
    output_root: Path,
    category: str,
    split: str = "train",
    num_points: int = 15000,
    max_models: int | None = None,
    seed: int = 0,
) -> list[Path]:
    """Convert one ShapeNetCore category into root/synset/split/*.npy layout."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    synset = category_to_synset(category)
    obj_files = find_category_obj_files(raw_root, category)
    if max_models is not None:
        obj_files = obj_files[:max_models]

    output_dir = output_root / synset / split
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, obj_path in enumerate(obj_files):
        vertices, faces = load_obj_mesh(obj_path)
        points = sample_mesh_points(vertices, faces, num_points=num_points, seed=seed + index)
        model_id = obj_path.parent.parent.name if obj_path.parent.name == "models" else obj_path.stem
        output_path = output_dir / f"{model_id}.npy"
        np.save(output_path, points)
        written.append(output_path)
    return written
