#!/usr/bin/python3
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import imageio
imageio.plugins.freeimage.download() # download the OpenEXR backend (if not already installed)

inp_dir = "Custom"
[img_dir, mask_dir, contour_dir] = ["img", "mask", "contour"]

processing_size = (256, 256)

points_per_side_incl_start_corner = 4
contour_pts = 4 * points_per_side_incl_start_corner

def save_uncompressed_npy(path, arr):
    with open(path, "wb") as f:
        np.save(f, arr)


def save_png_img_in_processing_size(inp_path, out_path):
    inp = cv2.imread(inp_path, cv2.IMREAD_ANYCOLOR)
    out = cv2.resize(inp, processing_size)
    cv2.imwrite(out_path, out)
    return out

def task(pairs):
    for (stem, out_img_dir, _, _, out_mask_dir, out_contour_dir) in pairs:
        save_png_img_in_processing_size(f"{inp_dir}/{img_dir}/{stem}.jpg", f"{out_img_dir}/{stem}.png")
        save_png_img_in_processing_size(f"{inp_dir}/{mask_dir}/{stem}.jpg", f"{out_mask_dir}/{stem}.png")
        shutil.copyfile(f"{inp_dir}/{contour_dir}/{stem}.npy", f"{out_contour_dir}/{stem}.npy")


def pairs(out_dirs):
    pairs = []

    paths = []
    paths.extend(glob(f"{inp_dir}/{img_dir}/*.jpg"))

    for path in paths:
        stem = Path(path).stem
        pairs.append((stem, *out_dirs))

    return pairs