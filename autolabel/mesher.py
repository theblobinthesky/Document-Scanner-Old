#!/usr/bin/python3
import os
import subprocess
import sys
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

# Based on:
# https://github.com/cdcseacave/openMVS/blob/master/MvgMvsPipeline.py

OPENMVG_BIN = "/usr/local/bin"
OPENMVS_BIN = OPENMVG_BIN

CAMERA_SENSOR_DB_FILE = "sensor_width_camera_database.txt"
CAMERA_SENSOR_DB_DIRECTORY = "/home/erik/Downloads/ReleaseV2.0.Rainbow-Trout.WindowsBinaries_VS2019/exif/sensor_width_database/"

def make_steps(input_dir, matches_dir, camera_file_params, reconstruction_dir, mvs_dir):
    return [
        ["Intrinsics analysis",
         os.path.join(OPENMVG_BIN, "openMVG_main_SfMInit_ImageListing"),
         ["-i", input_dir, "-o", matches_dir, "-d", camera_file_params]],
        ["Compute features",
         os.path.join(OPENMVG_BIN, "openMVG_main_ComputeFeatures"),
         ["-i", f"{matches_dir}/sfm_data.json", "-o", matches_dir, "-m", "SIFT"]],
        ["Compute pairs",
         os.path.join(OPENMVG_BIN, "openMVG_main_PairGenerator"),
         ["-i", f"{matches_dir}/sfm_data.json", "-o", f"{matches_dir}/pairs.bin"]],
        ["Compute matches",
         os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
         ["-i", f"{matches_dir}/sfm_data.json", "-p", f"{matches_dir}/pairs.bin", "-o", f"{matches_dir}/matches.putative.bin", "-n", "AUTO"]],
        ["Filter matches",
         os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
         ["-i", f"{matches_dir}/sfm_data.json", "-m", f"{matches_dir}/matches.putative.bin", "-o", f"{matches_dir}/matches.f.bin"]],
        ["Incremental reconstruction",
         os.path.join(OPENMVG_BIN, "openMVG_main_SfM"),
         ["-i", f"{matches_dir}/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir, "-s", "INCREMENTAL"]],
        ["Convert to json",
         os.path.join(OPENMVG_BIN, "openMVG_main_ConvertSfM_DataFormat"),
         ["-i", f"{reconstruction_dir}/sfm_data.bin", "-o", f"{reconstruction_dir}/sfm_data.json", "-V", "-I", "-E", "-C"]],
        ["Export to openMVS",
         os.path.join(OPENMVG_BIN, "openMVG_main_openMVG2openMVS"),
         ["-i", f"{reconstruction_dir}/sfm_data.bin", "-o", f"{mvs_dir}/scene.mvs", "-d", f"{mvs_dir}/images"]],
        ["Densify point cloud",
         os.path.join(OPENMVS_BIN, "MVSDensifyPointCloud"),
         ["scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "1", "--number-views", "8", "-w", f"\"{mvs_dir}\""]],
        ["Reconstruct the mesh",
         os.path.join(OPENMVS_BIN, "MVSReconstructMesh"),
         ["scene_dense.mvs", "-w", f"\"{mvs_dir}\""]]
        # ["Refine the mesh",
        #  os.path.join(OPENMVS_BIN, "MVSRefineMesh"),
        #  ["scene_dense_mesh.mvs", "--scales", "1", "--gradient-step", "25.05", "-w", f"\"{mvs_dir}\""]]
        ]


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

def rm_dirs(dirs):
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)

def solve_in_3d(input_dir, output_dir):
    input_dir = Path(input_dir).absolute()
    output_dir = Path(output_dir).absolute()

    if not os.path.exists(input_dir):
        sys.exit(f"input path not found: {input_dir}")

    rm_dirs([output_dir])

    reconstruction_dir, matches_dir = os.path.join(output_dir, "sfm"), os.path.join(output_dir, "matches") 
    mvs_dir = os.path.join(output_dir, "mvs")
    camera_file_params = os.path.join(CAMERA_SENSOR_DB_DIRECTORY, CAMERA_SENSOR_DB_FILE)
    make_dirs([output_dir, matches_dir, reconstruction_dir, mvs_dir])

    steps = make_steps(input_dir, matches_dir, camera_file_params, reconstruction_dir, mvs_dir)


    pbar = tqdm(total=len(steps))

    for [name, path, args] in steps:
        pbar.set_description(f"3d solver ({name})")

        try:
            cmdline = [path] + args
            pipes = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            std_out, std_err = pipes.communicate()
            pipes.wait()
            
            if pipes.returncode != 0 and std_err != b"":
                print("--- STANDARD OUTPUT ---")
                print(std_out.decode())
                print("--- STANDARD ERROR ---")
                print(std_err.decode())
                exit()

        except KeyboardInterrupt:
            sys.exit('\r\nProcess canceled by user, all files remains')

        pbar.update()

    pbar.close()


import open3d as o3d
import open3d.core as o3c
import matplotlib.pyplot as plt
import json

def test(input_dir, output_dir):
    sfm_data_path = f"{output_dir}/sfm/sfm_data.json"
    scene_dense_path = f"{output_dir}/mvs/scene_dense_mesh.ply"

    with open(sfm_data_path, "r") as f:
        data = json.load(f)

    views = []
    intrinsics, extrinsics = {}, {}


    for extrinsic in data["extrinsics"]:
        key = extrinsic["key"]

        view = extrinsic["value"]

        rotation = np.array(view["rotation"]).reshape((3, 3))
        rotation = np.linalg.inv(rotation)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3,:3] = rotation
        extrinsic_matrix[:3,3] = np.array(view["center"])

        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics[key] = extrinsic_matrix


    if False:
        # Load the sfm_data.json file
        sfm_data_path = f"{output_dir}/sfm/sfm_data.json"
        with open(sfm_data_path, "r") as f:
            sfm_data = json.load(f)

        # Create an Open3D visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the dense point cloud to the visualization
        pcd = o3d.io.read_point_cloud(f"{output_dir}/mvs/scene_dense.ply")
        vis.add_geometry(pcd)

        pcd = o3d.io.read_triangle_mesh(scene_dense_path)
        vis.add_geometry(pcd)

        # Add the camera poses to the visualization
        for view in sfm_data["views"]:
            pose_id = view["value"]["ptr_wrapper"]["data"]["id_pose"]
            extrinsic = extrinsics[pose_id]
            
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            camera.transform(extrinsic)
            vis.add_geometry(camera)

        # Run the visualization
        vis.run()
        vis.destroy_window()
        exit()

    for view in data["views"]:
        view = view["value"]["ptr_wrapper"]["data"]
        filename = view["filename"]
        size = (view["width"], view["height"])
        id_pose = view["id_pose"]
        id_intrinsic = view["id_intrinsic"]

        views.append((filename, size, id_pose, id_intrinsic))


    for intrinsic in data["intrinsics"]:
        key = intrinsic["key"]

        view = intrinsic["value"]["ptr_wrapper"]["data"]
        size = (view["width"], view["height"])
        focal_length = view["focal_length"]
        principal_point = view["principal_point"]
        distortion_coeff_k3 = view["disto_k3"]        

        intrinsic_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ], np.float32)

        intrinsics[key] = (size, intrinsic_matrix, distortion_coeff_k3)


    mesh = o3d.io.read_triangle_mesh(scene_dense_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    for i in range(len(views)):
        filename, size, pose_id, id_intrinsic = views[i]
        size, intrinsic_matrix, distortion_coeff_k3 = intrinsics[id_intrinsic]
        extrinsic_matrix = extrinsics[pose_id]
        
        intrinsic_matrix = o3c.Tensor(intrinsic_matrix)
        extrinsic_matrix = o3c.Tensor(extrinsic_matrix) 

        rays = scene.create_rays_pinhole(intrinsic_matrix, extrinsic_matrix, size[0], size[1])
        ans = scene.cast_rays(rays)

        depth = np.rot90(ans['t_hit'].numpy(), k=3)[:, :, np.newaxis]
        img = cv2.imread(f"{input_dir}/{filename}")
        img = img.astype("float32") / 255.0

        plt.imshow(depth)
        plt.title(filename)        
        plt.show()