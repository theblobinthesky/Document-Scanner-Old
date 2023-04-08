#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import imageio
imageio.plugins.freeimage.download() # download the OpenEXR backend (if not already installed)
from tqdm import tqdm
import concurrent.futures
import random

dir_pairs = [
    ("Doc3d/img/1", "Doc3d_64x64/img/1", "Doc3d_64x64/img_masked/1", "Doc3d_64x64/contour/1", "Doc3d/lines/1", "Doc3d_64x64/lines/1", "Doc3d/bm/1exr", "Doc3d_64x64/bm/1exr", "Doc3d/uv/1", "Doc3d_64x64/uv/1"),
    ("Doc3d/img/2", "Doc3d_64x64/img/2", "Doc3d_64x64/img_masked/2", "Doc3d_64x64/contour/2", "Doc3d/lines/2", "Doc3d_64x64/lines/2", "Doc3d/bm/2exr", "Doc3d_64x64/bm/2exr", "Doc3d/uv/2", "Doc3d_64x64/uv/2"),
    ("Doc3d/img/3", "Doc3d_64x64/img/3", "Doc3d_64x64/img_masked/3", "Doc3d_64x64/contour/3", "Doc3d/lines/3", "Doc3d_64x64/lines/3", "Doc3d/bm/3exr", "Doc3d_64x64/bm/3exr", "Doc3d/uv/3", "Doc3d_64x64/uv/3"),
    ("Doc3d/img/4", "Doc3d_64x64/img/4", "Doc3d_64x64/img_masked/4", "Doc3d_64x64/contour/4", "Doc3d/lines/4", "Doc3d_64x64/lines/4", "Doc3d/bm/4exr", "Doc3d_64x64/bm/4exr", "Doc3d/uv/4", "Doc3d_64x64/uv/4"),
    ("MitIndoor_64x64/img", None, None, None, None, None, None, None, None, None)
]

blank_heatmap_path = "blank_heatmap.npy"

processing_size = (256, 256)
downscale_size = (64, 64)

points_per_side_incl_start_corner = 4
contour_pts = 4 * points_per_side_incl_start_corner

min_max_scale = (1.0, 1.0) # (1, 1.1)
min_max_rot = (0.0, 360.0)

augment_factor = 3

def make_dirs(dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)

def random_between(between):
    return between[0] + random.random() * (between[1] - between[0])

def generate_random_transform():
    return random_between(min_max_scale), random_between(min_max_rot)

def get_mat_from_transform(size, scale, rot):
    size = np.array(size, np.int32)
    mat = cv2.getRotationMatrix2D(size / 2, rot, scale)
    return mat

def transform_image(img, mat):
    return cv2.warpAffine(img, mat, img.shape[:2])

def transform_points(pts, mat):
    b, _ = pts.shape
    ones = np.ones(shape=(b, 1))
    pts_ones = np.hstack([pts, ones])
    return mat.dot(pts_ones.T).T

def get_tl_corners_idx(corners):
    ys = [(y, i) for i, y in enumerate(corners[:, 1])]
    ys.sort()

    y_idx = [i for _, i in ys[:2]]

    xs = [(corners[i, 0], i) for i in y_idx]
    xs.sort()

    return xs[0][1]


def save_uncompressed_npy(path, arr):
    with open(path, "wb") as f:
        np.save(f, arr)


def generate_heatmap(cx, cy, confidence, size):
    map = np.zeros((cx.size, size[0], size[1]), np.float32)

    for i in range(cx.size):
        map[i, cx[i], cy[i]] = 1.0

    return map


def task(pairs):
    for (name, img_dir, img_down_dir, img_masked_down_dir, contour_dir, lines_dir, lines_down_dir, bm_dir, bm_down_dir, uv_dir, uv_down_dir) in pairs:
        img_path = f"{img_dir}/{name}.png"
        lines_path = f"{lines_dir}/{name}.png"
        bm_path = f"{bm_dir}/{name}.exr"
        uv_path = f"{uv_dir}/{name}.exr"

        unaug_img = cv2.imread(img_path)
        unaug_img = cv2.resize(unaug_img, processing_size)

        unaug_lines = cv2.imread(lines_path)
        unaug_lines = cv2.resize(unaug_lines, processing_size)

        unaug_bm = cv2.imread(bm_path, cv2.IMREAD_UNCHANGED)
        unaug_bm = cv2.resize(unaug_bm, processing_size)
        unaug_bm = unaug_bm[:, :, 1:3].astype("float32")

        unaug_uv = cv2.imread(uv_path, cv2.IMREAD_UNCHANGED)
        unaug_uv = cv2.resize(unaug_uv, processing_size)
        unaug_uv = unaug_uv.astype("float32")


        h, w, _ = unaug_bm.shape

        def interp(bm, start, end, i):
            start, end = np.array(start), np.array(end)
            t = float(i) / float(points_per_side_incl_start_corner)
        
            pt = (end * t + start * (1.0 - t)).astype("int32")
            return bm[pt[0], pt[1]].reshape(-1, 2)

        unaug_contour = []
                        
        # top
        for i in range(points_per_side_incl_start_corner):
            unaug_contour.append(interp(unaug_bm, (0, 0), (w - 1, 0), i))

        # right
        for i in range(points_per_side_incl_start_corner):
            unaug_contour.append(interp(unaug_bm, (w - 1, 0), (w - 1, h - 1), i))
                
        # bottom
        for i in range(points_per_side_incl_start_corner):
            unaug_contour.append(interp(unaug_bm, (w - 1, h - 1), (0, h - 1), i))
                
        # left
        for i in range(points_per_side_incl_start_corner):
            unaug_contour.append(interp(unaug_bm, (0, h - 1), (0, 0), i))

        unaug_contour = np.concatenate(unaug_contour, axis=0)


        for augment_index in range(augment_factor):
            img_down_path = f"{img_down_dir}/{name}_aug{augment_index}.png"
            img_masked_down_path = f"{img_masked_down_dir}/{name}_aug{augment_index}.png"
            contour_path = f"{contour_dir}/{name}_aug{augment_index}.npy" 
            lines_down_path = f"{lines_down_dir}/{name}_aug{augment_index}.png"
            bm_down_path = f"{bm_down_dir}/{name}_aug{augment_index}.exr"
            uv_down_path = f"{uv_down_dir}/{name}_aug{augment_index}.exr"
            
            # apply random transform
            scale, rot = generate_random_transform()

            mat = get_mat_from_transform(processing_size, scale, rot)
            img = transform_image(unaug_img, mat)
            img_down = cv2.resize(img, downscale_size)
            
            lines = transform_image(unaug_lines, mat)
            lines_down = cv2.resize(lines, downscale_size)

            mask_down = lines_down[:, :, 2][:, :, np.newaxis]
            mask_down = (mask_down == 255).astype("uint8") * 255
            mask_down = mask_down.repeat(3, axis=2)          
            img_masked_down = cv2.bitwise_and(img_down, mask_down)
            
            uv = transform_image(unaug_uv, mat)
            uv_down = cv2.resize(uv, downscale_size)

            # todo: understand -rot
            mat = get_mat_from_transform((1, 1), scale, -rot)
            contour = transform_points(unaug_contour, mat)

            # figure out which point is tl and adjust accordingly
            corners = np.array([
                contour[0 * points_per_side_incl_start_corner],
                contour[1 * points_per_side_incl_start_corner],
                contour[2 * points_per_side_incl_start_corner],
                contour[3 * points_per_side_incl_start_corner]
            ], np.float32)

            tl_idx = get_tl_corners_idx(corners)
            contour = np.roll(contour, shift=-tl_idx * points_per_side_incl_start_corner, axis=0)

            # generate confidences
            is_contour_out_of_view = np.bitwise_or((contour < 0.0).any(1), (contour > 1.0).any(1))
            is_contour_confident = (1.0 - is_contour_out_of_view.astype("float32"))
            is_contour_confident = is_contour_confident[:, np.newaxis]

            cx, cy = contour[:, 0], contour[:, 1]
            cx = np.round(downscale_size[0] * cx, 0).astype("int32")
            cy = np.round(downscale_size[1] * cy, 0).astype("int32")
            heatmap = generate_heatmap(cx, cy, is_contour_confident, downscale_size)


            bm_points = unaug_bm.reshape((-1, 2))
            bm_points = transform_points(bm_points, mat)
            bm_points = bm_points.reshape((*processing_size, 2)).astype("float32")
            bm_down = cv2.resize(bm_points, downscale_size)
            
            zeros = np.zeros((*downscale_size, 1), np.float32)
            bm_down = np.concatenate([zeros, bm_down], axis=2)

            cv2.imwrite(img_down_path, img_down)
            cv2.imwrite(img_masked_down_path, img_masked_down)
            cv2.imwrite(lines_down_path, lines_down)
            imageio.imsave(bm_down_path, bm_down)
            imageio.imsave(uv_down_path, uv_down)

            save_uncompressed_npy(contour_path, contour)

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for dir_pair in dir_pairs:
        img_dir, img_down_dir, img_masked_down_dir, contour_dir, lines_dir, lines_down_dir, bm_dir, bm_down_dir, uv_dir, uv_down_dir = dir_pair

        if contour_dir == None or bm_dir == None:
            feature_map = np.zeros((contour_pts, *downscale_size), np.float32)
            save_uncompressed_npy(blank_heatmap_path, feature_map)
            continue

        make_dirs([*dir_pair])

        paths = []
        paths.extend(glob(f"{img_dir}/*.png"))
        paths.extend(glob(f"{img_dir}/*.jpg"))

        for path in paths:
            name = Path(path).stem
            pairs.append((
                name, 
                img_dir, img_down_dir, img_masked_down_dir, contour_dir,
                lines_dir, lines_down_dir,
                bm_dir, bm_down_dir, uv_dir, uv_down_dir
            ))


    chunkedPairs = chunk(pairs, 64)

    futures = []
    for pairs in chunkedPairs:
        futures.append(executor.submit(task, pairs))
    
    for future in tqdm(futures):
        future.result()