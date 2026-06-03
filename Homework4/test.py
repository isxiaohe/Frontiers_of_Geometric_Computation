import os
import argparse
import time
import numpy as np
import torch
from skimage.measure import marching_cubes
import trimesh
from tqdm import tqdm

from model import SDFMLP, GaussianFourierFeatureTransform
from dataset import RecDataset


def extract_mesh(model, bbox_min, bbox_max, resolution=256, device="cpu", level=0.005):
    """
    在bbox内生成均匀网格，推理SDF，使用Marching Cubes提取mesh。
    每z-slice作为一个batch，batch_size = resolution^2。
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

    batch_size = resolution ** 2
    sdf_values = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    with torch.no_grad():
        for zi in tqdm(range(resolution), desc="Slices", ncols=100):
            xx, yy = np.meshgrid(x, y, indexing="ij")
            zz = np.full_like(xx, z[zi])
            grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

            slice_sdf = []
            for i in range(0, grid_points.shape[0], batch_size):
                batch_pts = torch.from_numpy(grid_points[i:i + batch_size]).to(device)
                batch_sdf = model(batch_pts).cpu().numpy()
                slice_sdf.append(batch_sdf)

            sdf_values[:, :, zi] = np.concatenate(slice_sdf, axis=0).reshape(resolution, resolution)

    # Marching Cubes在指定level处提取mesh
    sdf_min = sdf_values.min()
    sdf_max = sdf_values.max()
    print(f"SDF field range: [{sdf_min:.4f}, {sdf_max:.4f}]")

    if not (sdf_min <= level <= sdf_max):
        print(f"WARNING: level={level} is outside SDF range. Auto-adjusting to midpoint.")
        level = (sdf_min + sdf_max) / 2.0
        print(f"Using iso-level: {level:.4f}")

    try:
        vertices, faces, normals, _ = marching_cubes(sdf_values, level=level)
    except ValueError as e:
        print(f"ERROR: Marching Cubes failed: {e}")
        print("This usually means the model is not trained enough (SDF field doesn't cross zero).")
        # Return an empty mesh as fallback
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        return mesh

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
    dataset = RecDataset(data_dir, mode=args.mode, sample_size=1)
    bbox_min = dataset.bbox_min
    bbox_max = dataset.bbox_max

    # 加载模型
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        model_tag = '_' + '_'.join(args.ckpt.split('/')[-1].split('.')[0].split('_')[1:])
        breakpoint()
    else:
        model_tag = f"{args.mode}{'_fourier' if args.use_fourier else ''}"
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.uid}_{model_tag}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 支持两种checkpoint格式
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            model_args = ckpt.get("args", {})
        else:
            # 直接保存的state_dict
            state_dict = ckpt
            model_args = {}
    else:
        # 直接保存的模型对象（罕见）
        if hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
            model_args = {}
        else:
            state_dict = ckpt
            model_args = {}

    # 从checkpoint恢复模型参数
    hidden_dim = model_args.get("hidden_dim", 512)
    num_layers = model_args.get("num_layers", 10)
    activation = model_args.get("activation", "relu")
    skip_layers = model_args.get("skip_layers", [5])
    geometric_init = model_args.get("geometric_init", True)
    radius_init = model_args.get("radius_init", 1.0)
    use_fourier = model_args.get("use_fourier", False)
    mapping_size = model_args.get("mapping_size", 64)
    sigma = model_args.get("sigma", 5.0)

    if use_fourier:
        fourier_transform = GaussianFourierFeatureTransform(
            in_dim=3, mapping_size=mapping_size, sigma=sigma
        )
        model = SDFMLP(
            in_dim=2 * mapping_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            skip_layers=skip_layers,
            fourier_transform=fourier_transform,
            geometric_init=geometric_init,
            radius_init=radius_init,
        ).to(device)
    else:
        model = SDFMLP(
            in_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            skip_layers=skip_layers,
            geometric_init=geometric_init,
            radius_init=radius_init,
        ).to(device)

    model.load_state_dict(state_dict)
    print("Loaded checkpoint")

    # 提取mesh
    print(f"Extracting mesh with resolution {args.resolution}...")
    start_time = time.time()
    mesh = extract_mesh(
        model,
        bbox_min,
        bbox_max,
        resolution=args.resolution,
        device=device,
        level=args.level,
    )
    elapsed = time.time() - start_time
    print(f"Mesh extraction done in {elapsed:.1f}s")

    if args.clean_mesh:
        print("Cleaning mesh: keeping largest connected component...")
        components = mesh.split(only_watertight=False)
        if len(components) > 0:
            mesh = max(components, key=lambda c: len(c.vertices))
        print(f"Cleaned mesh - Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    else:
        print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # 保存结果
    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{args.uid}_{model_tag}.obj")
    mesh.export(result_path)
    print(f"Saved mesh to: {result_path}")

    # 如果指定了gt_path，加载并对比
    gt_path = os.path.join(data_dir, f"{args.uid}.obj")
    if os.path.exists(gt_path):
        gt_mesh = trimesh.load(gt_path, force="mesh")
        print(f"GT mesh - Vertices: {len(gt_mesh.vertices)}, Faces: {len(gt_mesh.faces)}")


def test_all(args):
    """测试所有shape。"""
    uids = sorted(
        [
            d
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ]
    )
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
    parser.add_argument(
        "--uid", type=str, default=None, help="指定测试单个shape的uid，不指定则测试所有"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["sdf", "surface", "mixed"],
        help="数据采样模式（与训练一致）",
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="Marching Cubes网格分辨率"
    )
    parser.add_argument(
        "--level", type=float, default=0.005, help="Marching Cubes等值面level"
    )
    parser.add_argument(
        "--use_fourier", action="store_true", help="是否加载Fourier feature模型"
    )
    parser.add_argument(
        "--clean_mesh", action="store_true", help="是否清理mesh，只保留最大连通组件"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="直接指定checkpoint文件路径（覆盖自动查找）"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="模型检查点目录"
    )
    parser.add_argument(
        "--result_dir", type=str, default="results", help="结果保存目录"
    )

    args = parser.parse_args()

    if args.uid is None:
        test_all(args)
    else:
        test(args)
