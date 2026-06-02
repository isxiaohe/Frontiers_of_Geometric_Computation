import os
import numpy as np
import trimesh


def load_mesh(path):
    """加载mesh文件。"""
    return trimesh.load(path, force="mesh")


def save_mesh(mesh, path):
    """保存mesh到文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def compute_chamfer_distance(pred_mesh, gt_mesh, num_samples=10000):
    """
    计算Chamfer Distance（双向最近邻距离的平均值）。
    """
    pred_pts = pred_mesh.sample(num_samples)
    gt_pts = gt_mesh.sample(num_samples)

    # pred -> gt
    dist_pred_gt = np.sqrt(((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=-1))
    min_pred_gt = dist_pred_gt.min(axis=1).mean()

    # gt -> pred
    dist_gt_pred = np.sqrt(((gt_pts[:, None, :] - pred_pts[None, :, :]) ** 2).sum(axis=-1))
    min_gt_pred = dist_gt_pred.min(axis=1).mean()

    return min_pred_gt + min_gt_pred


def compute_iou(pred_mesh, gt_mesh, num_samples=100000):
    """
    通过在空间采样点并判断内外，近似计算IoU。
    这里使用occupancy的方法：在bbox内随机采样点，用trimesh的contains判断内外。
    """
    # 合并bbox
    bbox_min = np.minimum(pred_mesh.bounds[0], gt_mesh.bounds[0])
    bbox_max = np.maximum(pred_mesh.bounds[1], gt_mesh.bounds[1])

    samples = np.random.uniform(bbox_min, bbox_max, size=(num_samples, 3))

    pred_occ = pred_mesh.contains(samples)
    gt_occ = gt_mesh.contains(samples)

    intersection = np.logical_and(pred_occ, gt_occ).sum()
    union = np.logical_or(pred_occ, gt_occ).sum()

    return intersection / union if union > 0 else 0.0
