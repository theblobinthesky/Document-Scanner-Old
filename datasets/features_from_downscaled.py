#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
from tqdm import tqdm
import concurrent.futures
import random

dir_pairs = [
    ("Doc3d/img/1", "Doc3d_64x64/img/1", "Doc3d_64x64/contour_feature/1", "Doc3d/lines/1", "Doc3d_64x64/lines/1", "Doc3d/bm/1exr"),
    ("Doc3d/img/2", "Doc3d_64x64/img/2", "Doc3d_64x64/contour_feature/2", "Doc3d/lines/2", "Doc3d_64x64/lines/2", "Doc3d/bm/2exr"),
    ("Doc3d/img/3", "Doc3d_64x64/img/3", "Doc3d_64x64/contour_feature/3", "Doc3d/lines/3", "Doc3d_64x64/lines/3", "Doc3d/bm/3exr"),
    ("Doc3d/img/4", "Doc3d_64x64/img/4", "Doc3d_64x64/contour_feature/4", "Doc3d/lines/4", "Doc3d_64x64/lines/4", "Doc3d/bm/4exr"),
    ("MitIndoor_64x64/img", None, None, None, None, None)
]

blank_contour_feature_path = "blank_contour_feature.npy"

processing_size = (256, 256)
downscale_size = (64, 64)

points_per_side_incl_start_corner = 4
feature_map_size = np.array([8, 8])
feature_depth = 4 + 4 * points_per_side_incl_start_corner * 3 + 1

min_max_scale = (1.0, 1.0) # (1, 1.1)
min_max_rot = (0.0, 0.0) # (0.0, 360.0)

augment_factor = 1

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

def task(pairs):
    for (name, img_dir, img_down_dir, feature_map_dir, lines_dir, lines_down_dir, bm_dir) in pairs:
        img_path = f"{img_dir}/{name}.png"
        lines_path = f"{lines_dir}/{name}.png"
        bm_path = f"{bm_dir}/{name}.exr"

        unaug_img = cv2.imread(img_path)
        unaug_img = cv2.resize(unaug_img, processing_size)

        unaug_lines = cv2.imread(lines_path)
        unaug_lines = cv2.resize(unaug_lines, processing_size)
        unaug_mask = unaug_lines[:, :, 2]

        unaug_bm = cv2.imread(bm_path, cv2.IMREAD_UNCHANGED)
        unaug_bm = cv2.resize(unaug_bm, processing_size)
        unaug_bm = unaug_bm[:, :, 1:3]
        
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
            feature_map_path = f"{feature_map_dir}/{name}_aug{augment_index}.npy" 
            lines_down_path = f"{lines_down_dir}/{name}_aug{augment_index}.png" 
            
            # apply random transform
            scale, rot = generate_random_transform()

            mat = get_mat_from_transform(unaug_img.shape[:2], scale, rot)
            img = transform_image(unaug_img, mat)
            img_down = cv2.resize(img, downscale_size)
            
            mat = get_mat_from_transform(unaug_lines.shape[:2], scale, rot)
            lines = transform_image(unaug_lines, mat)
            lines_down = cv2.resize(lines, downscale_size)
            
            mat = get_mat_from_transform(unaug_mask.shape[:2], scale, rot)
            mask = transform_image(unaug_mask, mat)
            mask = cv2.resize(mask, feature_map_size)
            mask = (mask == 255)

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

            # concat confidence in predictions
            is_contour_out_of_view = np.bitwise_or((contour < 0.0).any(1), (contour > 1.0).any(1))
            is_contour_confident = (1.0 - is_contour_out_of_view.astype("float32"))
            is_contour_confident = is_contour_confident[:, np.newaxis]

            contour = np.concatenate([contour, is_contour_confident], axis=-1)


            # calculate bounding box
            cx, cy = contour[:, 0], contour[:, 1]
            left, right = cx.min(), cx.max()
            top, bottom = cy.min(), cy.max()
            width, height = np.array([right - left], np.float32), np.array([bottom - top], np.float32)
            center = np.array([(left + right) / 2.0, (top + bottom) / 2.0], np.float32)

            # offset contour points based on center
            cx -= center[0]
            cy -= center[1]
            
            # contour points are relative to the bounding box for now
            # if width != 0.0:
            #     cx /= (width / 2.0)
            
            # if height != 0.0:
            #     cy /= (height / 2.0)

            # find grid cell responsible for prediction
            cell = np.floor(center * feature_map_size)
            cell_norm = cell / feature_map_size
            center_rel_to_cell = (center - cell_norm) * feature_map_size
            cell = cell.astype("int32")

            bbox = np.concatenate([center_rel_to_cell, width, height])

            # features
            one = np.array([1.0], np.float32)
            feature = np.concatenate([bbox, contour.reshape(-1), one])

            # replicate features into larger map
            feature_map = np.zeros((*feature_map_size, feature.size), dtype=np.float32)
            feature_map[cell[0], cell[1]] = feature

            w, h, f = feature_map.shape
            feature_map_rs = feature_map.reshape(-1, f)
            feature_map_id = mask.reshape(-1).nonzero()
            feature_map_rs[feature_map_id][:, -1] = 1.0

            assert feature_map.shape[-1] == feature_depth

            feature_map = feature_map.transpose((2, 0, 1))

            cv2.imwrite(img_down_path, img_down)
            cv2.imwrite(lines_down_path, lines_down)

            dict = {
                "feature_map": feature_map,
                "cell": cell
            }

            np.save(feature_map_path, dict)

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for (img_dir, img_down_dir, feature_map_dir, lines_dir, lines_down_dir, bm_dir) in dir_pairs:
        if feature_map_dir == None or bm_dir == None:
            feature_map = np.zeros((feature_depth, feature_map_size[0], feature_map_size[1]), np.float32)
            cell = np.zeros((2), np.int32)

            dict = {
                "feature_map": feature_map,
                "cell": cell
            }

            np.save(blank_contour_feature_path, dict)
            continue

        make_dirs([img_dir, img_down_dir, feature_map_dir, lines_dir, lines_down_dir, bm_dir])

        paths = []
        paths.extend(glob(f"{img_dir}/*.png"))
        paths.extend(glob(f"{img_dir}/*.jpg"))

        for path in paths:
            name = Path(path).stem
            pairs.append((
                name, 
                img_dir, img_down_dir, feature_map_dir,
                lines_dir, lines_down_dir,
                bm_dir
            ))

    chunkedPairs = chunk(pairs, 64)

    futures = []
    for pairs in chunkedPairs:
        futures.append(executor.submit(task, pairs))
    
    for future in tqdm(futures):
        future.result()