#!/usr/bin/python3
from glob import glob
import time
from mesher import solve_in_3d, track_points_onto_mesh
from support import get_best_segmentation, get_guided_segmentations, save_images_and_masks, get_largest_contour, get_simple_contour
import utils
import math
import traceback

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

    scans = utils.get_scans()
    for scan_dir in scans:
        data = utils.get_scan_data(scan_dir)
        paths = utils.get_unrotated_img_paths(data)

        n_cols = 3
        n_rows = int(math.ceil(len(paths) / float(n_cols)))

        for i, path in enumerate(paths):
            name = Path(path).name
            img = f"{utils.image_dir}/{name}"
            mask = f"{utils.mask_dir}/{name}"

            img, mask = cv2.imread(img), cv2.imread(mask)
            img = img.astype("float32") / 255.0
            mask = mask.astype("float32") / 255.0
            
            comp = 0.6 * img + 0.4 * mask
            
            plt.subplot(n_rows, n_cols, i + 1)

            if name == data["best_mask_name"]:
                plt.title(f"{name} (segmented)")
            else:
                plt.title(f"{name} (tracked)")

            plt.axis("off")
            plt.imshow(comp)

        plt.show()

    exit()


scans = utils.get_scans()
scans.sort()

for i, scan_dir in enumerate(scans):
    start_time = time.time()
    
    print()
    print(f"Working on {i + 1}/{len(scans)} '{scan_dir}'...")

    try:
        data = utils.get_scan_data(scan_dir)
        if not data["label_changed"]:
            print("Skipping since label didn't change...")
            continue

        scan_name = data["scan_name"]

        if "autolabel_complete" in data:
            print("Skipping 3d solve...")
        else:
            solve_in_3d(scan_dir, utils.output_dir)
            utils.move_3d_solve_and_unrotated_images(data)


        paths = utils.get_unrotated_img_paths(data)
        
        if "manulabel_name" in data:
            print("Skipping dense segmentation for manual mask...")

            best_name = data["manulabel_name"]
            best_path = f"{utils.label_dir}/{scan_name}/img/{best_name}"

            best_mask = utils.get_manual_mask(data)
            best_contour = get_largest_contour(best_mask)
            best_contour = get_simple_contour(best_contour, 0.002)
        else:
            best_path, best_contour, is_confident = get_best_segmentation(paths)
            data["best_mask_name"] = utils.get_filename(best_path)
            
            if is_confident: 
                print("Prediction is too unconfident.")
                continue

        contours = track_points_onto_mesh(paths, best_path, best_contour, data)
        masks = get_guided_segmentations(paths, contours)
        save_images_and_masks(paths, masks, utils.output_dir, utils.image_dir, utils.mask_dir)

        data["autolabel_complete"] = True
        data["label_changed"] = False
        utils.save_scan_data(data)
    except Exception as e:
        print("Autolabel threw an error:")
        traceback.print_exception(type(e), e, e.__traceback__)
        print("Skipping...")
        continue


    runtime = time.time() - start_time
    runtime /= 60.0

    print(f"Finished in {runtime:.2f}min")