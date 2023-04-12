import os
from pathlib import Path
import shutil
import json
import cv2
from glob import glob

scans_dir = "../datasets/Custom/scans"
label_dir = "../datasets/Custom/label"
image_dir = "../datasets/Custom/img"
mask_dir = "../datasets/Custom/mask"
data_name = "data.json"
output_dir = "output"

def get_scans():
    return [ f.path for f in os.scandir(scans_dir) if f.is_dir() ]

def get_scan_name_from_path(scan_dir):
    return scan_dir[len(scans_dir) + 1:]

def get_scan_data(scan_dir):
    scan_name = get_scan_name_from_path(scan_dir)
    scan_label_dir = f"{label_dir}/{scan_name}"
    data_path = f"{scan_label_dir}/{data_name}"

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f) 

        return data
    else:
        return {"scan_name": scan_name, "label_changed": True}

def get_scan_label_dir(data):
    scan_name = data["scan_name"]
    return f"{label_dir}/{scan_name}"

def move_3d_solve_and_unrotated_images(data):
    scan_label_dir = get_scan_label_dir(data)
    make_dirs([scan_label_dir])
    
    sfm_data_path = f"{output_dir}/sfm/sfm_data.json"
    scene_dense_path = f"{output_dir}/mvs/scene_dense_mesh.ply"

    shutil.copyfile(sfm_data_path, f"{scan_label_dir}/sfm_data.json")
    shutil.copyfile(scene_dense_path, f"{scan_label_dir}/scene_dense_mesh.ply")
    shutil.copytree(f"{output_dir}/mvs/images", f"{scan_label_dir}/img", dirs_exist_ok=True)

    # remove the useless files
    shutil.rmtree(output_dir)


def get_unrotated_img_paths(data):
    scan_name = data["scan_name"]
    paths = glob(f"{label_dir}/{scan_name}/img/*.jpg")

    return paths

def get_manual_mask(data):
    scan_name = data["scan_name"]
    manulabel_name = data["manulabel_name"]
    manulabel_path = f"{label_dir}/{scan_name}/{manulabel_name}"
    
    mask = cv2.imread(manulabel_path, cv2.IMREAD_GRAYSCALE)

    return mask

def save_scan_data(data):
    scan_name = data["scan_name"]
    data_path = f"{label_dir}/{scan_name}/{data_name}"

    with open(data_path, "w") as f:
        json.dump(data, f)


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            Path(dir).mkdir(parents=True, exist_ok=True)

def get_filename(path):
    return Path(path).name