#!/usr/bin/python3
import os
from glob import glob
import time
from mesher import solve_in_3d, track_points_onto_mesh
from support import get_best_segmentation, get_guided_segmentations, get_mask_from_contour

dir = "/media/shared/Projekte/DocumentScanner/datasets/Custom"
model_path = "models/main_seg.pth"
processing_size = 256
model_size = 64
mask_size_kernels = (15, 7)
mask_size_factors = (0.9, 1.1)
binarize_threshold = 0.8
max_ratio = 0.03
track_box_size = 51
hbs = track_box_size // 2
iters = 4

scans_dir = "data/scans/scan_0"
output_dir = "data/output"

scans = [x[0] for x in os.walk(scans_dir)]

for images_dir in scans:
    print()
    print(f"Working on '{images_dir}'...")

    start_time = time.time()

    solve_in_3d(images_dir, output_dir)

    paths = glob(f"{output_dir}/mvs/images/*.jpg")
    best_path, best_contour = get_best_segmentation(paths)
    contours = track_points_onto_mesh(paths, best_path, best_contour, output_dir)
    masks = get_guided_segmentations(paths, contours)

    runtime = time.time() - start_time
    runtime /= 60.0
    print(f"Finished in {runtime:.2f}min")

    import matplotlib.pyplot as plt
    for path, mask in masks:
        if path == best_path:
            plt.title(f"{path} (best path)")
        else:
            plt.title(f"{path} (tracked path)")

        plt.imshow(mask)
        plt.show()