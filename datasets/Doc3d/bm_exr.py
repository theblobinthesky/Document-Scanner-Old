#!/usr/bin/python3
from glob import glob
import h5py
import OpenEXR
from pathlib import Path
import numpy as np
from tqdm import tqdm
import concurrent.futures

dir_pairs = [("bm/3", "bm/3exr")]

def task(pairs):
    for (src, dst) in pairs:    
        mat = h5py.File(src)
        mat = mat['bm']
        mat = mat[:,:,:].astype('float32') / 448.0
        mat = np.transpose(mat, [0, 2, 1])
        mat = np.ascontiguousarray(mat)
        
        file = OpenEXR.OutputFile(dst, OpenEXR.Header(mat.shape[2], mat.shape[1]))
        file.writePixels({'R': mat[0], 'G': mat[1]})
        file.close()

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    pairs = []
    for (src_dir, dst_dir) in dir_pairs:
        paths = glob(f"{src_dir}/*.mat")

        for path in paths:
            name = Path(path).stem
            pairs.append((path, f"{dst_dir}/{name}.exr"))


    chunkedPairs = chunk(pairs, 32)
    futures = []
    for pairs in chunkedPairs:
        futures.append(executor.submit(task, pairs))
    
    for future in tqdm(futures):
        future.result()
