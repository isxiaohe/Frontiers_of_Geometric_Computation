"""PointNet++ classification models: SSG and MSG variants.

Architecture overview:

  SSG (Single-Scale Grouping):
    Input: (B, N, 3)
      → SA1: N→512,  r=0.2, k=32,  MLP [64,64,128]     → (B, 128, 512)
      → SA2: 512→128, r=0.4, k=64, MLP [128,128,256]    → (B, 256, 128)
      → SA3: 128→1,   global,      MLP [256,512,1024]    → (B, 1024, 1)
      → FC: 1024→512→256→40 (with dropout + BN + ReLU)

  MSG (Multi-Scale Grouping):
    Input: (B, N, 3) or (B, N, 6) with normals
      → SA1-MSG: N→512, radii [0.1,0.2,0.4], MLPs [[32,32,64],[64,64,128],[64,96,128]]
        → (B, 64+128+128, 512) = (B, 320, 512)
      → SA2-MSG: 512→128, radii [0.2,0.4,0.8], MLPs [[64,64,128],[128,128,256],[128,128,256]]
        → (B, 128+256+256, 128) = (B, 640, 128)
      → SA3: 128→1, global, MLP [256,512,1024]  (single-scale, group_all)
        → (B, 1024, 1)
      → FC: 1024→512→256→40

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on
Point Sets in a Metric Space", NeurIPS 2017.
"""
import torch
import torch.nn as nn
from .pointnet_util import PointNetSetAbstraction, PointNetSetAbstractionMsg


class PointNet2ClsSSG(nn.Module):
    """PointNet++ classifier with Single-Scale Grouping.

    Args:
        num_classes:   int, number of output classes (default 40 for ModelNet40).
        normal_channel: bool, if True, input has 6 channels (xyz + normals)
                       instead of 3 (xyz only).
    """

    def __init__(self, num_classes=40, normal_channel=False):
        super().__init__()
        in_channel = 6 if normal_channel else 3

        # ── Hierarchical encoder: 3 SA layers ──
        #
        # SA1: N → 512 points, ball radius 0.2, 32 neighbors
        #   Input features:  in_channel (3 or 6) + 3 (relative xyz) per group
        #   MLP: in_channel → 64 → 64 → 128
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128],
        )

        # SA2: 512 → 128 points, ball radius 0.4, 64 neighbors
        #   Input features: 128 (from SA1) + 3 (relative xyz)
        #   MLP: 128 → 128 → 128 → 256
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128, mlp=[128, 128, 256],
        )

        # SA3: 128 → 1 point (global), group ALL remaining points
        #   Input features: 256 (from SA2) + 3 (relative xyz)
        #   MLP: 256 → 256 → 512 → 1024
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256, mlp=[256, 512, 1024], group_all=True,
        )

        # ── Classification head ──
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6) — point cloud, optionally with normals.

        Returns:
            logits: (B, num_classes) — unnormalized class scores.

        Steps:
          1. Split xyz into coordinates (first 3) and optional normals.
             - xyz_coords = xyz[:, :, :3]        → (B, N, 3)
             - If normal_channel: points = xyz[:, :, 3:].transpose(1,2) → (B, 3, N)
             - Else: points = None
          2. Pass through SA layers sequentially:
             l1_xyz, l1_points = self.sa1(xyz_coords, points)
             l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
             l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
          3. Flatten global feature: x = l3_points.reshape(B, -1)
          4. FC head with BN + ReLU + Dropout:
             x = drop1(relu(bn1(fc1(x))))
             x = drop2(relu(bn2(fc2(x))))
             x = fc3(x)
          5. Return x
        """
        B = xyz.shape[0]

        # First SA layer: xyz as both coordinates AND features
        xyz_coords = xyz[:, :, :3].contiguous()
        if xyz.shape[-1] > 3:
            # normals present: features = normals
            points = xyz[:, :, 3:].transpose(1, 2).contiguous()
        else:
            # no normals: features = xyz itself
            points = xyz_coords.transpose(1, 2).contiguous()

        # Hierarchical SA
        l1_xyz, l1_points = self.sa1(xyz_coords, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # FC head
        x = l3_points.reshape(B, -1)                        # (B, 1024)
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))   # (B, 512)
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))   # (B, 256)
        x = self.fc3(x)                                     # (B, num_classes)
        return x


class PointNet2ClsMSG(nn.Module):
    """PointNet++ classifier with Multi-Scale Grouping.

    Uses MSG for the first two SA layers to handle non-uniform sampling density.
    The third SA layer is global (single-scale, group_all=True).

    Args:
        num_classes:   int, number of output classes.
        normal_channel: bool, if True, input includes normals (6 channels).
    """

    def __init__(self, num_classes=40, normal_channel=False):
        super().__init__()
        in_channel = 6 if normal_channel else 3

        # SA1-MSG: N → 512 points, 3 scales
        #   Scale 0: r=0.1, k=16,  MLP [32, 32, 64]
        #   Scale 1: r=0.2, k=32,  MLP [64, 64, 128]
        #   Scale 2: r=0.4, k=128, MLP [64, 96, 128]
        #   Output: 64 + 128 + 128 = 320 channels
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=in_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )

        # SA2-MSG: 512 → 128 points, 3 scales
        #   Input: 320 channels from SA1
        #   Scale 0: r=0.2, k=32,  MLP [64, 64, 128]
        #   Scale 1: r=0.4, k=64,  MLP [128, 128, 256]
        #   Scale 2: r=0.8, k=128, MLP [128, 128, 256]
        #   Output: 128 + 256 + 256 = 640 channels
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=320,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )

        # SA3: 128 → 1 point (global), single-scale
        #   Input: 640 channels from SA2
        #   MLP: 640 → 256 → 512 → 1024
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=640, mlp=[256, 512, 1024], group_all=True,
        )

        # Classification head (same as SSG)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6)

        Returns:
            logits: (B, num_classes)

        Steps: Same as SSG, but SA1 and SA2 are MSG layers.
          1. Split xyz / normals
          2. sa1 → sa2 → sa3
          3. FC head with BN + ReLU + Dropout
        """
        B = xyz.shape[0]

        # First SA layer: xyz as both coordinates AND features
        xyz_coords = xyz[:, :, :3].contiguous()
        if xyz.shape[-1] > 3:
            points = xyz[:, :, 3:].transpose(1, 2).contiguous()
        else:
            points = xyz_coords.transpose(1, 2).contiguous()

        # Hierarchical SA (MSG for first two layers)
        l1_xyz, l1_points = self.sa1(xyz_coords, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # FC head
        x = l3_points.reshape(B, -1)                        # (B, 1024)
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))   # (B, 512)
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))   # (B, 256)
        x = self.fc3(x)                                     # (B, num_classes)
        return x
