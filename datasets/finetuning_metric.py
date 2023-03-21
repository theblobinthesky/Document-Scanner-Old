#!/usr/bin/python3
from glob import glob
from pathlib import Path
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt

dir_items = [
    ("Doc3d_64x64/img/1", "Doc3d_64x64/lines/1"),
    ("Doc3d_64x64/img/2", "Doc3d_64x64/lines/2"),
    ("Doc3d_64x64/img/3", "Doc3d_64x64/lines/3"),
    ("Doc3d_64x64/img/4", "Doc3d_64x64/lines/4"), 
    ("MitIndoor_64x64/img", None),
]

metric = {}
metric_error = []

border_size = 7
border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size, border_size))
nudge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

pick_size = 4
picks_per_row = 2

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def task(items):
    for (img_path, lines_path) in items: 
        if lines_path == None:
            metric_error.append(img_path)
            continue

        img = cv2.imread(img_path)
        lines = cv2.imread(lines_path)
        mask = lines[:, :, 2]

        # calculate metric
        mask_small = cv2.erode(mask, border_kernel)
        mask_large = cv2.dilate(mask, border_kernel)
        mask_channel = cv2.bitwise_and(cv2.bitwise_not(mask_small), mask_large)

        mask_nudge_small = cv2.erode(mask, nudge_kernel)
        mask_nudge_large = cv2.dilate(mask, nudge_kernel)

        inner_channel = cv2.bitwise_and(img, cv2.bitwise_and(mask_channel, mask_nudge_small)[:, :, np.newaxis].repeat(3, axis=-1))
        outer_channel = cv2.bitwise_and(img, cv2.bitwise_and(mask_channel, cv2.bitwise_not(mask_nudge_large))[:, :, np.newaxis].repeat(3, axis=-1))
        
        inner, outer = inner_channel.mean(), outer_channel.mean()

        metric[img_path] = abs(inner - outer)


def chunk(list, chunk_size):
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]


with concurrent.futures.ThreadPoolExecutor(8) as executor:
    items = []
    for (img_dir, lines_dir) in dir_items:
        make_dir(img_dir)
    
        if lines_dir != None:
            make_dir(lines_dir)

        paths = []
        paths.extend(glob(f"{img_dir}/*.png"))
        paths.extend(glob(f"{img_dir}/*.jpg"))

        for path in paths:
            if lines_dir == None:
                items.append((path, None))
            else:
                path_obj = Path(path)
                name = path_obj.stem
                ext = path_obj.suffix[1:]
                items.append((path, f"{lines_dir}/{name}.{ext}"))


    chunkedItems = chunk(items, 32)
    futures = []
    for itemChunk in chunkedItems:
        futures.append(executor.submit(task, itemChunk))

    for future in tqdm(futures):
        future.result()

    min_metric, max_metric = min(metric.values()), max(metric.values())
    for key, value in metric.items():
        metric[key] = (value - min_metric) / max_metric

    # invert metric bc. it currently captures loss rather than gain
    for key, value in metric.items():
        metric[key] = 1.0 - metric[key]

    for error in metric_error:
        metric[error] = 1.0

    print(f"{min_metric=}, {max_metric=}")

    with open("finetuning_metric.npy", "wb") as file:
        np.save(file, np.array(metric))

sorted_items = [(key, value) for key, value in sorted(metric.items(), key=lambda item: item[1])]

for i, (key, value) in enumerate(sorted_items):
    if value == 0.0: continue
    else:
        sorted_items = sorted_items[i:]
        break

for i, (key, value) in enumerate(reversed(sorted_items)):
    if value == 1.0: continue
    else:
        sorted_items = sorted_items[:-i]
        break

start_pick = sorted_items[:pick_size]
end_pick = sorted_items[-pick_size:]
picks = start_pick + end_pick
total_picks = len(picks)

fig = plt.figure(figsize=(25, 25))

for i, (path, metric) in enumerate(picks):
    ax = plt.subplot(total_picks, picks_per_row, i + 1)
    plt.title(f"{metric:.4f}")
    ax.imshow(cv2.imread(path), aspect="equal")
    ax.axis('off')

plt.show()