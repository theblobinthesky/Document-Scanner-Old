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
    ("Doc3d_64x64/img/4", "Doc3d_64x64/flatten/4", "Doc3d/bm/4exr")
]

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def task(pairs):
    for (flatten_path, bm_path) in pairs:
        if bm_path == None:
            coords = np.zeros((9), np.float32)
        else:
            bm = cv2.imread(bm_path, cv2.IMREAD_UNCHANGED)
            bm = bm[:, :, 1:3]

            h, w, _ = bm.shape

            tl, tr = bm[0, 0], bm[h - 1, 0]            
            bl, br = bm[0, w - 1], bm[h - 1, w - 1]

            coords = np.concatenate([np.array([1.0], np.float32), tl, tr, bl, br])

        file = OpenEXR.OutputFile(flatten_path, OpenEXR.Header(1, coords.shape[0]))
        file.writePixels({'R': coords})
        file.close()

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for (img_dir, flatten_dir, bm_dir) in dir_pairs:
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