#!/usr/bin/python3
import os
from glob import glob
import time
from mesher import solve_in_3d, track_points_onto_mesh
from support import get_best_segmentation, get_guided_segmentations, save_images_and_masks
import utils

dir = "/media/shared/Projekte/DocumentScanner/datasets/Custom"
model_path = "models/main_seg.pth"
processing_size = 256
model_size = 64
mask_size_kernels = (15, 7)
mask_size_factors = (0.9, 1.1)
binarize_threshold = 0.8
min_ratio = 0.03
track_box_size = 51
hbs = track_box_size // 2
iters = 4

if False:
    import cv2
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    images = glob(f"{image_dir}/*.jpg")
    for image in images:
        name = Path(image).name
        mask = f"{mask_dir}/{name}"

        image, mask = cv2.imread(image), cv2.imread(mask)
        image = cv2.bitwise_and(image, mask)
        
        plt.title(name)
        plt.imshow(image)
        plt.show()

    exit()


scans = utils.get_scans()
for scan_dir in scans:
    print()
    print(f"Working on '{scan_dir}'...")

    start_time = time.time()
    scan_name = utils.get_scan_name_from_path(scan_dir)
    scan_output_dir = f"{utils.output_dir}/{scan_name}"

    if os.path.exists(scan_output_dir):
        print("Skipping 3d solve...")
    else:
        solve_in_3d(scan_dir, scan_output_dir)

    paths = glob(f"{scan_output_dir}/mvs/images/*.jpg")
    best_path, best_contour, is_confident = get_best_segmentation(paths)
    if is_confident: 
        print("Prediction is too unconfident.")
        continue

    contours = track_points_onto_mesh(paths, best_path, best_contour, scan_output_dir)
    masks = get_guided_segmentations(paths, contours)
    save_images_and_masks(paths, masks, scan_output_dir, utils.image_dir, utils.mask_dir)

    runtime = time.time() - start_time
    runtime /= 60.0
    print(f"Finished in {runtime:.2f}min")