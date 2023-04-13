#!/usr/bin/python3
import os
import subprocess
import sys
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
import open3d as o3d
import open3d.core as o3c
import json

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
         ["-i", f"{matches_dir}/sfm_data.json", "-o", matches_dir, "-m", "SIFT", "-p", "ULTRA"]],
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
        ]


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            Path(dir).mkdir(parents=True)

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

def raycast_contour_onto_mesh(scene_dense_path, views, intrinsics, extrinsics, path, contour):
    mesh = o3d.io.read_triangle_mesh(scene_dense_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    filename = Path(path).name
    size, pose_id, id_intrinsic = views[filename]
    size, intrinsic_matrix, distortion_coeff_k3 = intrinsics[id_intrinsic]
    extrinsic_matrix = extrinsics[pose_id]
        
    intrinsic_matrix = o3c.Tensor(intrinsic_matrix)
    extrinsic_matrix = o3c.Tensor(extrinsic_matrix) 

    rays = scene.create_rays_pinhole(intrinsic_matrix, extrinsic_matrix, size[0], size[1])
    ans = scene.cast_rays(rays)


    ray_pos = rays[:, :, :3]
    ray_dir = rays[:, :, 3:]

    ray_len = ans["t_hit"].numpy()
    is_infinite = np.bitwise_not(np.isfinite(ray_len))
    ray_len = ray_len[:, :, np.newaxis]

    contour = contour.astype("int32")
    xs, ys = contour[:, 0], contour[:, 1]

    pt_invalid = is_infinite[ys, xs]
    for i in range(len(pt_invalid)):
        if pt_invalid[i]:
            l_idx, r_idx = i - 1, i + 1
            while pt_invalid[l_idx]: l_idx = l_idx - 1 # Find the first valid element before i
            while pt_invalid[r_idx]: r_idx = r_idx + 1 # Find the first valid element after i

            # todo: this is a stupid way to tell, use actual uv coordinates later!
            l_pt, c_pt, r_pt = contour[l_idx], contour[i], contour[r_idx]
            t = np.abs((c_pt - l_pt) / (r_pt - l_pt)).max()
            ray_len[c_pt[1], c_pt[0]] = (1.0 - t) * ray_len[l_pt[1], l_pt[0]] + t * ray_len[r_pt[1], r_pt[0]]

    pts = ray_pos[ys, xs] + ray_dir[ys, xs] * ray_len[ys, xs]
    pts = pts.numpy()
    
    return pts


def track_points_onto_mesh(paths, best_path, best_contour, scan_label_dir):
    sfm_data_path = f"{scan_label_dir}/sfm_data.json"
    scene_dense_path = f"{scan_label_dir}/scene_dense_mesh.ply"

    with open(sfm_data_path, "r") as f:
        data = json.load(f)

    views, intrinsics, extrinsics = {}, {}, {}

    for view in data["views"]:
        view = view["value"]["ptr_wrapper"]["data"]
        filename = view["filename"]
        size = (view["width"], view["height"])
        id_pose = view["id_pose"]
        id_intrinsic = view["id_intrinsic"]

        views[filename] = (size, id_pose, id_intrinsic)


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


    for extrinsic in data["extrinsics"]:
        key = extrinsic["key"]

        view = extrinsic["value"]

        rotation = np.array(view["rotation"]).reshape((3, 3))
        rotation = np.linalg.inv(rotation)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3,:3] = rotation
        extrinsic_matrix[:3, 3] = np.array(view["center"])

        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics[key] = extrinsic_matrix


    contour_points = raycast_contour_onto_mesh(scene_dense_path, views, intrinsics, extrinsics, best_path, best_contour)

    contours = []
    for path in paths:
        name = Path(path).name
        size, id_pose, id_intrinsic = views[name]
        size, intrinsic_matrix, distortion_coeff_k3 = intrinsics[id_intrinsic]
        extrinsic_matrix = extrinsics[id_pose][:3, :4]
        matrix = intrinsic_matrix @ extrinsic_matrix

        projected_points = []
        for contour_point in contour_points:
            contour_point = np.concatenate([contour_point, np.array([1.0])])
            point = matrix @ contour_point
            point = np.array([point[0] / point[2], point[1] / point[2]])
            projected_points.append(point)

        contours.append(projected_points)

    return np.array(contours), np.array(contour_points)

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