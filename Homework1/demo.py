"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
from tqdm import tqdm 

import argparse

import fusion


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", action="store_true", help="Whether to integrate color into voxel volume (optional)")
    args = parser.parse_args()
    
    
    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    
    # vol_bnds[:, 0]: minimum bounds of the voxel volume along x, y, z
    # vol_bnds[:, 1]: maximum bounds of the voxel volume along x, y, z
    vol_bnds = np.zeros((3, 2))
    
    for i in tqdm(range(n_imgs)):
        
        # Read depth image 
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i), -1).astype(float)
        # depth is saved in 16-bit PNG in millimeters
        depth_im /= 1000.  
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0  
        
        # Read camera pose, a 4x4 rigid transformation matrix
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  
        
        #######################    Task 1    #######################
        #  Convert depth image to world coordinates
        view_frust_pts = fusion.cam_to_world(
            depth_im, cam_intr, cam_pose,
            export_pc=(i%100 == 0),  # export pointcloud only for the first frame,
            export_path="pcd/pointcloud_%06d.ply"%(i)  # to save disk space
        )
        # TODO: Update voxel volume bounds `vol_bnds`
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.min(view_frust_pts, axis=0))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.max(view_frust_pts, axis=0))
        ############################################################
    print("Volume bounds:", vol_bnds)

    # Initialize TSDF voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in tqdm(range(n_imgs)):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

        # Integrate observation into voxel volume
        color_im = cv2.imread("data/frame-%06d.color.jpg"%(i))  # read color image (optional)
        tsdf_vol.integrate(cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB), depth_im, cam_intr, cam_pose, obs_weight=1., color_integration=args.color)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    #######################    Task 4    #######################
    # TODO: Extract mesh from voxel volume, save and visualize it
    tsdf = tsdf_vol.tsdf_vol.reshape(tsdf_vol.vol_dims)
    # marching cubes
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes(tsdf, level=0)
    verts_world = verts * tsdf_vol.voxel_size + tsdf_vol.vol_bnds[:, 0]
    
    # get vertex colors by nearest neighbor interpolation (optional)
    # Convert 3D vertex coordinates to linear indices for accessing flattened color volume
    verts_int = verts.astype(int)
    verts_idx = (verts_int[:, 0] * tsdf_vol.vol_dims[1] * tsdf_vol.vol_dims[2] +
                 verts_int[:, 1] * tsdf_vol.vol_dims[2] +
                 verts_int[:, 2])
    verts_idx = np.clip(verts_idx, 0, len(tsdf_vol.color_vol) - 1)
    vertex_colors = (tsdf_vol.color_vol[verts_idx] * 255).clip(0, 255).astype(np.uint8)
    
    

    # save mesh as .ply file
    with open(f"tsdf_mesh{('_color' if args.color else '')}.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(verts_world)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(faces)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(verts_world, vertex_colors):
            if args.color:
                f.write("{} {} {} {} {} {}\n".format(v[0], v[1], v[2], c[0], c[1], c[2]))
            else:
                f.write("{} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("3 {} {} {}\n".format(face[0], face[1], face[2]))
    
    ############################################################

