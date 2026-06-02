import os
import argparse
import time
import numpy as np
import torch
from skimage.measure import marching_cubes
import trimesh

from model import SDFMLP, FourierFeatureMLP
from dataset import SDFDataset


def extract_mesh(model, bbox_min, bbox_max, resolution=128, device="cpu", batch_size=65536):
    """
    在bbox内生成均匀网格，推理SDF，使用Marching Cubes提取mesh。
    """
    model.eval()

    # padding 10%
    extent = bbox_max - bbox_min
    bbox_min = bbox_min - 0.1 * extent
    bbox_max = bbox_max + 0.1 * extent

    # 生成均匀网格点
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    # batch推理
    sdf_values = []
    with torch.no_grad():
        for i in range(0, grid_points.shape[0], batch_size):
            batch_pts = torch.from_numpy(grid_points[i:i + batch_size]).to(device)
            batch_sdf = model(batch_pts).cpu().numpy()
            sdf_values.append(batch_sdf)

    sdf_values = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)

    # Marching Cubes在sdf=0处提取mesh
    try:
        vertices, faces, normals, _ = marching_cubes(sdf_values, level=0)
    except ValueError:
        # 如果level=0不在范围内，尝试level=0.0但用allow_degenerate
        vertices, faces, normals, _ = marching_cubes(sdf_values, level=0)

    # 将顶点坐标从voxel空间映射回世界坐标
    scale = (bbox_max - bbox_min) / (resolution - 1)
    vertices = vertices * scale + bbox_min

    # 使用trimesh导出
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return mesh


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(args.data_root, args.uid)
    dataset = SDFDataset(data_dir)
    bbox_min = dataset.bbox_min
    bbox_max = dataset.bbox_max

    # 加载模型
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{args.mode}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt.get("args", {})

    # 从checkpoint恢复模型参数
    hidden_dim = model_args.get("hidden_dim", 256)
    num_layers = model_args.get("num_layers", 8)
    activation = model_args.get("activation", "relu")
    skip_layers = model_args.get("skip_layers", [4])
    mapping_size = model_args.get("mapping_size", 10)
    sigma = model_args.get("sigma", 10.0)

    if args.mode == "base":
        model = SDFMLP(
            in_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            skip_layers=skip_layers,
        ).to(device)
    elif args.mode == "fourier":
        model = FourierFeatureMLP(
            in_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            skip_layers=skip_layers,
            mapping_size=mapping_size,
            sigma=sigma,
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # 提取mesh
    print(f"Extracting mesh with resolution {args.resolution}...")
    start_time = time.time()
    mesh = extract_mesh(
        model,
        bbox_min,
        bbox_max,
        resolution=args.resolution,
        device=device,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start_time
    print(f"Mesh extraction done in {elapsed:.1f}s")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # 保存结果
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.uid}_{args.mode}.obj")
    mesh.export(result_path)
    print(f"Saved mesh to: {result_path}")

    # 如果指定了gt_path，加载并对比
    gt_path = os.path.join(data_dir, f"{args.uid}.obj")
    if os.path.exists(gt_path):
        gt_mesh = trimesh.load(gt_path, force="mesh")
        print(f"GT mesh - Vertices: {len(gt_mesh.vertices)}, Faces: {len(gt_mesh.faces)}")


def test_all(args):
    """测试所有shape。"""
    uids = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    print(f"Found {len(uids)} shapes: {uids}")
    for uid in uids:
        print(f"\n{'='*60}")
        print(f"Testing shape: {uid}")
        print(f"{'='*60}")
        args.uid = uid
        test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Neural SDF per shape")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--uid", type=str, default=None, help="指定测试单个shape的uid，不指定则测试所有")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "fourier"], help="测试模式")
    parser.add_argument("--resolution", type=int, default=128, help="Marching Cubes网格分辨率")
    parser.add_argument("--batch_size", type=int, default=65536, help="推理批次大小")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="模型检查点目录")
    parser.add_argument("--result_dir", type=str, default="results", help="结果保存目录")

    args = parser.parse_args()

    if args.uid is None:
        test_all(args)
    else:
        test(args)
