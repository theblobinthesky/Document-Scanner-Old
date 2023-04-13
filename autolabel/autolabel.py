#!/usr/bin/python3
from glob import glob
import time
from mesher import solve_in_3d, track_points_onto_mesh
import support
from support import get_best_segmentation, get_guided_segmentations, get_standard_contours_from_contours, save_images_masks_and_contours, get_largest_contour, get_simple_contour
import math
import traceback

dir = "/media/shared/Projekte/DocumentScanner/datasets/Custom"

scans = support.get_scans()
scans.sort()

if False:
    import cv2
    from pathlib import Path
    import matplotlib.pyplot as plt

    for scan_dir in scans:
        data = support.get_scan_data(scan_dir)
        
        if not "best_mask_name" in data:
            print("Skipping. Please run autolabel on this scan first.")
            continue

        paths = support.get_unrotated_img_paths(data)

        n_cols = 3
        n_rows = int(math.ceil(len(paths) / float(n_cols)))
        plt.figure(data["scan_name"], figsize=(25, 25))

        for i, path in enumerate(paths):
            name = Path(path).name
            img = f"{support.image_dir}/{name}"
            mask = f"{support.mask_dir}/{name}"

            img, mask = cv2.imread(img), cv2.imread(mask)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


for i, scan_dir in enumerate(scans):
    start_time = time.time()
    
    print()
    print(f"Working on {i + 1}/{len(scans)} '{scan_dir}'...")

    try:
        data = support.get_scan_data(scan_dir)
        if not data["label_changed"] and False:
            print("Skipping since label didn't change...")
            continue

        scan_name = data["scan_name"]

        if "autolabel_complete" in data:
            print("Skipping 3d solve...")
        else:
            print("lol no")
            continue
            solve_in_3d(scan_dir, support.output_dir)
            support.move_3d_solve_and_unrotated_images(data)


        paths = support.get_unrotated_img_paths(data)
        
        if "manulabel_name" in data:
            print("Skipping dense segmentation because of manual mask...")

            best_name = data["manulabel_name"]
            best_path = f"{support.label_dir}/{scan_name}/img/{best_name}"

            best_mask = support.get_manual_mask(data)
            best_contour = get_largest_contour(best_mask)
            best_contour = get_simple_contour(best_contour, 0.0012)
        else:
            best_path, best_contour, is_confident = get_best_segmentation(paths)
            data["best_mask_name"] = support.get_filename(best_path)
            
            if is_confident: 
                print("Prediction is too unconfident.")
                continue

        scan_label_dir = support.get_scan_label_dir(data)
        projected_contours, contour_3d = track_points_onto_mesh(paths, best_path, best_contour, scan_label_dir)
        projected_contours, masks = get_guided_segmentations(paths, projected_contours)
        contours = get_standard_contours_from_contours(projected_contours, contour_3d, masks[0].shape[0:2][::-1])
        save_images_masks_and_contours(paths, masks, contours, data)

        data["autolabel_complete"] = True
        data["label_changed"] = False
        support.save_scan_data(data)
        
    except Exception as e:
        print("Autolabel threw an error:")
        traceback.print_exception(type(e), e, e.__traceback__)
        print("Skipping...")
        continue


    runtime = time.time() - start_time
    runtime /= 60.0

    print(f"Finished in {runtime:.2f}min")