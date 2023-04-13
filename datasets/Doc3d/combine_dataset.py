#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import imageio
imageio.plugins.freeimage.download() # download the OpenEXR backend (if not already installed)

inp_dir = "Doc3d"
[img_dir, bm_dir, uv_dir] = ["img", "bm", "uv"]
subdirs = ["1", "2", "3", "4"]

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


def save_exr_img_in_processing_size(inp_path, out_path):
    inp = cv2.imread(inp_path, cv2.IMREAD_UNCHANGED)
    out = cv2.resize(inp, processing_size)
    imageio.imsave(out_path, out)
    return out


def task(pairs):
    for (name, inp_img_dir, inp_bm_dir, inp_uv_dir, out_img_dir, out_bm_dir, out_uv_dir, out_mask_dir, out_contour_dir) in pairs:
        save_png_img_in_processing_size(f"{inp_img_dir}/{name}.png", f"{out_img_dir}/{name}.png")
        unaug_bm = save_exr_img_in_processing_size(f"{inp_bm_dir}/{name}.exr", f"{out_bm_dir}/{name}.exr")
        unaug_uv = save_exr_img_in_processing_size(f"{inp_uv_dir}/{name}.exr", f"{out_uv_dir}/{name}.exr")  

        unaug_mask = (unaug_uv[:, :, 0] * 255).astype("uint8")
        unaug_mask = cv2.resize(unaug_mask, processing_size)
        cv2.imwrite(f"{out_mask_dir}/{name}.png", unaug_mask)
        
        def interp(bm, start, end, i):
            start, end = np.array(start), np.array(end)
            t = float(i) / float(points_per_side_incl_start_corner)
        
            pt = (end * t + start * (1.0 - t)).astype("int32")
            return bm[pt[0], pt[1]].reshape(-1, 2)

        unaug_bm = unaug_bm[:, :, 1:3]
        h, w, _ = unaug_bm.shape
        unaug_contour = []
        unaug_contour.extend([interp(unaug_bm, (0, 0), (w - 1, 0), i) for i in range(points_per_side_incl_start_corner)]) # top
        unaug_contour.extend([interp(unaug_bm, (w - 1, 0), (w - 1, h - 1), i) for i in range(points_per_side_incl_start_corner)]) # right
        unaug_contour.extend([interp(unaug_bm, (w - 1, h - 1), (0, h - 1), i) for i in range(points_per_side_incl_start_corner)]) # bottom
        unaug_contour.extend([interp(unaug_bm, (0, h - 1), (0, 0), i) for i in range(points_per_side_incl_start_corner)]) # left
        unaug_contour = np.array(unaug_contour).reshape((-1, 2))

        save_uncompressed_npy(f"{out_contour_dir}/{name}.npy", unaug_contour)


def pairs(out_dirs):
    pairs = []

    for subdir in subdirs:
        inp_img_dir = f"{inp_dir}/{img_dir}/{subdir}"
        paths = []
        paths.extend(glob(f"{inp_img_dir}/*.png"))


        for path in paths:
            name = Path(path).stem
            pairs.append((name, inp_img_dir, f"{inp_dir}/{bm_dir}/{subdir}", f"{inp_dir}/{uv_dir}/{subdir}", *out_dirs))

    return pairs