#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import concurrent.futures

src_dir = 'img/1'
uv_dir = 'uv/1'
dst_dir = 'img_masked/1'

paths = glob(f"{src_dir}/*.png")

def task(paths):
    for path in paths:    
        name = Path(path).stem

        img = cv2.imread(path)
        mask = cv2.imread(f"{uv_dir}/{name}.exr")
    
        cv2.imwrite(f"{dst_dir}/{name}.png", mask[:, :, 0][:, :, np.newaxis] * img)
        
    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    chunkedPaths = chunk(paths, 32)
    futures = []
    for paths in chunkedPaths:
        futures.append(executor.submit(task, paths))

    for future in tqdm(futures):
        future.result()
