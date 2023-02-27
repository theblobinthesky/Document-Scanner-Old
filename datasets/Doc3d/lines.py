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

dir_items = [("img/1", "uv/1", "bm/1exr", "lines/1"), ("img/2", "uv/2", "bm/2exr", "lines/2"), ("img/3", "uv/3", "bm/3exr", "lines/3")]
padding = 5

def exr_loader(path, ndim=3):
    file = OpenEXR.InputFile(path)

    dw = file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    def load_channel(name):
        C = file.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.frombuffer(C, np.float32)
        C = np.reshape(C, size)
        return C

    if ndim == 1:
        return np.transpose(load_channel('R'), [1, 2, 0])
    elif ndim == 3:
        channels = [load_channel(c)[np.newaxis, :] for c in ['R', 'G', 'B']]
        return np.transpose(np.concatenate(channels, axis=0), [1, 2, 0])
    else:
        print("incorrect number of channels.")
        exit()


def task(items):
    for (img_path, uv_path, bm_path, out_path) in items:    
        img = cv2.imread(path)
        uv = exr_loader(uv_path)
        mask = (uv[:, :, 2] * 255).astype("uint8")[:,:,np.newaxis]
        uv = uv[:, :, 0:2]
        bm = exr_loader(bm_path)[:, :, 0:2]

        out = cv2.remap(img, bm, None, interpolation=cv2.INTER_CUBIC)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = out.astype("uint8")
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)
        out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))

        contours, _ = cv2.findContours(out, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = np.zeros(out.shape, np.uint8)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ar = float(h) / float(w)
            
            if ar > 5.0 and w < 15.0 and h > 50.0:
                mx = x + w // 2
                cv2.line(lines, (mx, y + padding), (mx, y + h - 2 * padding), 255, 2)

        uv = 1.0 - uv        
        uv[:, :, 0] *= lines.shape[1]
        uv[:, :, 1] *= lines.shape[0]
        lines = np.transpose(lines, [1, 0])
        lines = cv2.remap(lines, uv, None, interpolation=cv2.INTER_AREA)[:,:,np.newaxis]
        
        out = np.concatenate([lines, lines, mask], axis=-1)
        
        cv2.imwrite(out_path, out)


def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    items = []
    for (img_dir, uv_dir, bm_dir, out_dir) in dir_items:
        paths = glob(f"{img_dir}/*.png")

        for path in paths:
            name = Path(path).stem
            items.append((path, f"{uv_dir}/{name}.exr", f"{bm_dir}/{name}.exr", f"{out_dir}/{name}.png"))


    chunkedItems = chunk(items, 32)
    futures = []
    for itemChunk in chunkedItems:
        futures.append(executor.submit(task, itemChunk))

    for future in tqdm(futures):
        future.result()