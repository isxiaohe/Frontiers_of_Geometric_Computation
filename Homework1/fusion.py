# Copyright (c) 2018 Andy Zeng

import numpy as np
from skimage import measure
import trimesh

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self.voxel_size  # truncation on SDF

        
        #######################    Task 2    #######################
        # TODO: build voxel grid coordinates and initiailze volumn attributes
        # Initialize voxel volume
        self.vol_dims = np.round((vol_bnds[:, 1] - vol_bnds[:, 0]) / self.voxel_size).astype(int)
        self.tsdf_vol = np.ones(self.vol_dims).astype(np.float32).flatten()  # tsdf values are initialized to truncation distance (i.e. 1) by convention
        # for computing the cumulative moving average of weights per voxel
        self.weight_vol = np.zeros(self.vol_dims).astype(np.float32).flatten()
        # Set voxel grid coordinates
        x, y, z = np.meshgrid(
            np.arange(self.vol_dims[0]), np.arange(self.vol_dims[1]), np.arange(self.vol_dims[2]), indexing='ij'
        )
        self.vox_coords = np.vstack((x.flatten(), y.flatten(), z.flatten())).T  # (N, 3) coordinates of voxel centers
        self.color_vol = np.zeros((self.vol_dims[0] * self.vol_dims[1] * self.vol_dims[2], 3), dtype=np.float32)  # for storing integrated color values (optional)
        ############################################################

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1., color_integration=False):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            color_im (ndarray): A color image of shape (H, W, 3).
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. 
        """

        #######################    Task 2    #######################
        # breakpoint()
        # TODO: Convert voxel grid coordinates to pixel coordinates
        vox_world = self.vox_coords * self.voxel_size + self.vol_bnds[:, 0]  # (N, 3) world coordinates of voxel centers
        vox_world_hom = np.hstack((vox_world, np.ones((vox_world.shape[0], 1))))  # (N, 4) homogeneous world coordinates
        vox_cam = (np.linalg.inv(cam_pose) @ vox_world_hom.T).T  # (N, 4) homogeneous camera coordinates
        vox_cam = vox_cam[:, :3]
        pix_z = vox_cam[:, 2]
        pix = (cam_intr @ vox_cam.T).T  # (N, 3) pixel coordinates in homogeneous form
        pix = np.round(pix[:, :2] / pix_z[:, None])
        # TODO: Eliminate pixels outside depth images
        valid_pix = np.logical_and(
            np.logical_and(pix[:, 0] >= 0, pix[:, 0] < depth_im.shape[1]),
            np.logical_and(pix[:, 1] >= 0, pix[:, 1] < depth_im.shape[0])
        )
        pix = pix[valid_pix]
        pix_z = pix_z[valid_pix]
        vox_cam = vox_cam[valid_pix]
        vox_world = vox_world[valid_pix]
        # TODO: Sample depth values
        depth_val = depth_im[pix[:, 1].astype(int), pix[:, 0].astype(int)]
        ############################################################
        
        #######################    Task 3    #######################
        # TODO: Compute TSDF for current frame
        depth_diff = depth_val - pix_z
        valid_pts = depth_diff >= -self.trunc_margin  # consider voxels within truncation distance from the surface (i.e. depth value)
        tsdf = np.clip(depth_diff / self.trunc_margin, -1, 1)
        tsdf = tsdf[valid_pts]
        vox_world = vox_world[valid_pts]
        vox_cam = vox_cam[valid_pts]
        pix = pix[valid_pts]
        ############################################################

        #######################    Task 4    #######################
        # TODO: Integrate TSDF into voxel volume
        # Get the actual indices of valid voxels
        # breakpoint()
        valid_pix_idx = np.where(valid_pix)[0]  # indices that pass pixel bounds check
        final_valid_idx = valid_pix_idx[valid_pts]  # indices that also pass truncation check
        w_old = self.weight_vol[final_valid_idx]
        w_new = w_old + obs_weight
        self.tsdf_vol[final_valid_idx] = (self.tsdf_vol[final_valid_idx] * w_old + tsdf * obs_weight) / w_new
        self.weight_vol[final_valid_idx] = w_new
        ############################################################
        
        #######################    Bonus    #######################
        # TODO: Integrate color into voxel volume (optional)
        if color_integration:
            color_val = color_im[pix[:, 1].astype(int), pix[:, 0].astype(int)] / 255.0  # normalize color values to [0, 1]
            self.color_vol[final_valid_idx] = (self.color_vol[final_valid_idx] * w_old[:, None] + color_val * obs_weight) / w_new[:, None]
        #############################################################


def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False, export_path="pointcloud.ply"):
    """Get 3D point cloud from depth image and camera pose
    
    Args:
        depth_im (ndarray): Depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        export_pc (bool): Whether to export pointcloud to a PLY file.
        
    Returns:
        world_pts (ndarray): The 3D point cloud of shape (N, 3).
    """
    
    #######################    Task 1    #######################
    # TODO: Convert depth image to world coordinates
    img_to_cam = np.linalg.inv(cam_intr)
    # cam_to_world = np.linalg.inv(cam_pose)
    # x is row index, y is column index
    x, y = np.meshgrid(np.arange(depth_im.shape[0]), np.arange(depth_im.shape[1]), indexing='ij')
    z = depth_im.flatten()
    x = x.flatten()
    y = y.flatten()
    cam_pts = img_to_cam @ np.vstack((y * z, x * z, z))
    cam_pts = np.vstack((cam_pts, np.ones((1, cam_pts.shape[1]))))
    world_pts = cam_pose @ cam_pts
    world_pts = world_pts[:3, :].T 
    ############################################################
    
    if export_pc:
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export(export_path)
    
    return world_pts
