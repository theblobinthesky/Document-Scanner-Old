#!/usr/bin/python3
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import shutil

from Custom.combine_dataset import pairs as custom_pairs, task as custom_task
from Doc3d.combine_dataset import pairs as doc3d_pairs, task as doc3d_task
from finetuning_metric import make_finetuning_metric

execution_pairs = [("Custom", custom_pairs, custom_task)]
                   # ("Doc3d", doc3d_pairs, doc3d_task)]
[img_dir, bm_dir, uv_dir, mask_dir, contour_dir] = ["img", "bm", "uv", "mask", "contour"]
out_dir = "Combined"
max_pairs = 100 # 6400

def rm_dirs(dirs):
    for dir in dirs:
        shutil.rmtree(dir, ignore_errors=True)

def make_dirs(dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)

out_dirs = [f"{out_dir}/{img_dir}", f"{out_dir}/{bm_dir}", f"{out_dir}/{uv_dir}", f"{out_dir}/{mask_dir}", f"{out_dir}/{contour_dir}"]
rm_dirs(out_dirs)
make_dirs(out_dirs)
    

def chunk(list, chunk_size):
    chunks = [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]

    if len(list) % chunk_size != 0:
        i = (len(list) // chunk_size) * chunk_size
        chunks.append(list[i:])

    return chunks


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    for (name, pairs, task) in execution_pairs:
        pairs = pairs(out_dirs)
        
        new_length = min(max_pairs, len(pairs))
        max_pairs -= new_length

        chunkedPairs = chunk(pairs[:new_length], 64)

        futures = []
        for pairs in chunkedPairs:
            futures.append(executor.submit(task, pairs))
        
        for future in tqdm(futures, f"Combine {name} dataset"):
            future.result()


make_finetuning_metric()