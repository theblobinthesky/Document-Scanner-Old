#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import concurrent.futures

dir_items = [("img/1", "uv/1", "img_masked/1"), ("img/2", "uv/2", "img_masked/2"), ("img/3", "uv/3", "img_masked/3")] 

def task(items):
    for (img_path, uv_path, out_path) in items:
        img = cv2.imread(img_path)
        mask = cv2.imread(uv_path)
        mask = mask[:, :, 0][:, :, np.newaxis]
    
        cv2.imwrite(out_path, mask * img)
        
    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    items = []
    for (img_dir, uv_dir, out_dir) in dir_items:
        paths = glob(f"{img_dir}/*.png")

        for path in paths:
            name = Path(path).stem
            items.append((path, f"{uv_dir}/{name}.exr", f"{out_dir}/{name}.png"))


    chunkedItems = chunk(items, 32)
    futures = []
    for itemChunk in chunkedItems:
        futures.append(executor.submit(task, itemChunk))

    for future in tqdm(futures):
        future.result()