#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
from tqdm import tqdm
import concurrent.futures

dir_pairs = [
    ("Doc3d_64x64/img/1", "Doc3d_64x64/contour_feature/1", "Doc3d_64x64/lines/1", "Doc3d/bm/1exr"),
    ("Doc3d_64x64/img/2", "Doc3d_64x64/contour_feature/2", "Doc3d_64x64/lines/2", "Doc3d/bm/2exr"),
    ("Doc3d_64x64/img/3", "Doc3d_64x64/contour_feature/3", "Doc3d_64x64/lines/3", "Doc3d/bm/3exr"),
    ("Doc3d_64x64/img/4", "Doc3d_64x64/contour_feature/4", "Doc3d_64x64/lines/4", "Doc3d/bm/4exr"),
    # ("MitIndoor_64x64/img", None, None, None)
]

blank_flatten_path = "blank_flatten.exr"

points_per_side_incl_start_corner = 4
feature_map_size = np.array([8, 8])
feature_depth = 4 + 4 * points_per_side_incl_start_corner * 3 + 1

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def task(pairs):
    for (feature_map_path, lines_path, bm_path) in pairs:
        mask = cv2.imread(lines_path)[:, :, 2]
        mask = cv2.resize(mask, feature_map_size)
        mask = (mask == 255)

        bm = cv2.imread(bm_path, cv2.IMREAD_UNCHANGED)
        bm = bm[:, :, 1:3]

        h, w, _ = bm.shape

        def interp(bm, start, end, i):
            start, end = np.array(start), np.array(end)
            t = float(i) / float(points_per_side_incl_start_corner)
        
            pt = (end * t + start * (1.0 - t)).astype("int32")
            one = np.array([1.0], np.float32)
            return np.concatenate([bm[pt[0], pt[1]], one], axis=-1).reshape(1, 3)

        contour = []
                      
        # top
        for i in range(points_per_side_incl_start_corner):
            contour.append(interp(bm, (0, 0), (w - 1, 0), i))

        # right
        for i in range(points_per_side_incl_start_corner):
            contour.append(interp(bm, (w - 1, 0), (w - 1, h - 1), i))
            
        # bottom
        for i in range(points_per_side_incl_start_corner):
            contour.append(interp(bm, (w - 1, h - 1), (0, h - 1), i))
            
        # left
        for i in range(points_per_side_incl_start_corner):
            contour.append(interp(bm, (0, h - 1), (0, 0), i))

        contour = np.concatenate(contour, axis=0)

        # calculate bounding box
        cx, cy = contour[:, 0], contour[:, 1]
        left, right = cx.min(), cx.max()
        top, bottom = cy.min(), cy.max()
        width, height = np.array([right - left], np.float32), np.array([bottom - top], np.float32)
        center = np.array([(left + right) / 2.0, (top + bottom) / 2.0], np.float32)

        # offset contour points based on center
        cx -= center[0]
        cy -= center[1]

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

        dict = {
            "feature_map": feature_map,
            "cell": cell
        }

        np.save(feature_map_path, dict)

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for (img_dir, feature_map_dir, lines_dir, bm_dir) in dir_pairs:
        if feature_map_dir == None or bm_dir == None:
        #     flatten = np.zeros((1), np.float32)
        #     save_npy(blank_flatten_path, flatten)
            continue

        make_dir(feature_map_dir)

        paths = []
        paths.extend(glob(f"{img_dir}/*.png"))
        paths.extend(glob(f"{img_dir}/*.jpg"))

        for path in paths:
            name = Path(path).stem
            pairs.append((f"{feature_map_dir}/{name}.npy", f"{lines_dir}/{name}.png", f"{bm_dir}/{name}.exr"))

    chunkedPairs = chunk(pairs, 64)
    futures = []
    for pairs in chunkedPairs:
        futures.append(executor.submit(task, pairs))
    
    for future in tqdm(futures):
        future.result()