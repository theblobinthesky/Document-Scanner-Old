#!/usr/bin/python3
from glob import glob
import h5py
import OpenEXR
from pathlib import Path
import numpy as np
from tqdm import tqdm
import concurrent.futures

src_dir = 'bm/1'
dst_dir = 'bm/1exr'

paths = glob(f"{src_dir}/*.mat")

def task(paths):
    for path in paths:    
        mat = h5py.File(path)
        mat = mat['bm']
        mat = mat[:,:,:].astype('float32')
        print(mat)
        
        name = Path(path).stem
        file = OpenEXR.OutputFile(f"{dst_dir}/{name}.exr", OpenEXR.Header(mat.shape[2], mat.shape[1]))
        file.writePixels({'R': mat[0], 'G': mat[1]})
        file.close()

    
def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    chunkedPaths = chunk(paths, 32)
    futures = []
    for paths in chunkedPaths:
        futures.append(executor.submit(task, paths))

    for future in tqdm(futures):
        future.result()
