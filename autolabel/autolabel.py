#!/usr/bin/python3
from glob import glob
import cv2
from tqdm import tqdm
import time
from mesher import solve_in_3d, test

dir = "/media/shared/Projekte/DocumentScanner/datasets/Custom"
model_path = "models/main_seg_2.pth"
processing_size = 256
model_size = 64
mask_size_kernels = (15, 7)
mask_size_factors = (0.9, 1.1)
binarize_threshold = 0.8
max_ratio = 0.03
track_box_size = 51
hbs = track_box_size // 2
iters = 4

images_dir = "data/images/new"
output_dir = "data/output"

print()
print(f"Working on '{images_dir}'...")

start_time = time.time()

paths = glob(f"{images_dir}/*.jpg")

# solve_in_3d(images_dir, output_dir)
test(images_dir, output_dir)

for path in paths:
    frame = cv2.imread(path)
    width, height, _ = frame.shape


    # contours_enum = do_dense_segmentation(cap, frames)
    # fused_track = do_fused_track(cap, contours_enum)



runtime = time.time() - start_time
runtime /= 60.0
print(f"Finished in {runtime:.2f}min")