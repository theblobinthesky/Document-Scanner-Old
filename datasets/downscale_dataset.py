#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import Imath
import OpenEXR
import cv2
from tqdm import tqdm
import concurrent.futures

out_size = (64, 64)

dir_items = [
    ("Doc3d/img/1", "Doc3d_64x64/img/1"), ("Doc3d/img/2", "Doc3d_64x64/img/2"), 
    ("Doc3d/img/3", "Doc3d_64x64/img/3"), ("Doc3d/img/4", "Doc3d_64x64/img/4"),
    ("Doc3d/lines/1", "Doc3d_64x64/lines/1"), ("Doc3d/lines/2", "Doc3d_64x64/lines/2"), 
    ("Doc3d/lines/3", "Doc3d_64x64/lines/3"), ("Doc3d/lines/4", "Doc3d_64x64/lines/4"),
]

def task(items):
    for (inp_path, out_path) in items:    
        inp = cv2.imread(inp_path)
        out = cv2.resize(inp, out_size)
        cv2.imwrite(out_path, out)


def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    items = []
    for (inp_dir, out_dir) in dir_items:
        paths = []
        paths.extend(glob(f"{inp_dir}/*.png"))
        paths.extend(glob(f"{inp_dir}/*.jpg"))

        for path in paths:
            path_obj = Path(path)
            name = path_obj.stem
            ext = path_obj.suffix
            items.append((path, f"{out_dir}/{name}.{ext}"))


    chunkedItems = chunk(items, 32)
    futures = []
    for itemChunk in chunkedItems:
        futures.append(executor.submit(task, itemChunk))

    for future in tqdm(futures):
        future.result()