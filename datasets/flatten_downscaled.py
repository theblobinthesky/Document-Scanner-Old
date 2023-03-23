#!/usr/bin/python3
from glob import glob
import h5py
import OpenEXR
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
from tqdm import tqdm
import concurrent.futures

dir_pairs = [
    ("Doc3d_64x64/img/1", "Doc3d_64x64/flatten/1", "Doc3d/bm/1exr"),
    ("Doc3d_64x64/img/2", "Doc3d_64x64/flatten/2", "Doc3d/bm/2exr"),
    ("Doc3d_64x64/img/3", "Doc3d_64x64/flatten/3", "Doc3d/bm/3exr"),
    ("Doc3d_64x64/img/4", "Doc3d_64x64/flatten/4", "Doc3d/bm/4exr"),
    ("MitIndoor_64x64/img", None, None)
]

blank_flatten_path = "blank_flatten.exr"

points_per_side_incl_start_corner = 4
flatten_size = points_per_side_incl_start_corner * 4 * 2 + 1

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def save_exr(path, flatten):
    file = OpenEXR.OutputFile(path, OpenEXR.Header(1, flatten.shape[0]))
    file.writePixels({'R': flatten})
    file.close()

def task(pairs):
    for (flatten_path, bm_path) in pairs:
        bm = cv2.imread(bm_path, cv2.IMREAD_UNCHANGED)
        bm = bm[:, :, 1:3]

        h, w, _ = bm.shape

        def interp(bm, start, end, i):
            start, end = np.array(start), np.array(end)
            t = float(i) / float(points_per_side_incl_start_corner)
        
            pt = (end * t + start * (1.0 - t)).astype("int32")
            return bm[pt[0], pt[1]]

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

        flatten = np.concatenate([np.array([1.0], np.float32), *contour])

    save_exr(flatten_path, flatten)

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for (img_dir, flatten_dir, bm_dir) in dir_pairs:
        if flatten_dir == None or bm_dir == None:
            flatten = np.zeros((flatten_size), np.float32)
            save_exr(blank_flatten_path, flatten)
            continue

        make_dir(flatten_dir)

        paths = []
        paths.extend(glob(f"{img_dir}/*.png"))
        paths.extend(glob(f"{img_dir}/*.jpg"))

        for path in paths:
            name = Path(path).stem
            
            if bm_dir == None:
                pairs.append((f"{flatten_dir}/{name}.exr", None))
            else:
                pairs.append((f"{flatten_dir}/{name}.exr", f"{bm_dir}/{name}.exr"))

    chunkedPairs = chunk(pairs, 32)
    futures = []
    for pairs in chunkedPairs:
        futures.append(executor.submit(task, pairs))
    
    for future in tqdm(futures):
        future.result()