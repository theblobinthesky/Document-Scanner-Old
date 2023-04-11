import os

scans_dir = "../datasets/Custom/scans"
annotation_subdir = "annotation"
image_dir = "../datasets/Custom/img"
mask_dir = "../datasets/Custom/mask"
output_dir = "output"

def get_scans():
    return [ f.path for f in os.scandir(scans_dir) if f.is_dir() ]

def get_scan_name_from_path(scan_dir):
    return scan_dir[len(scans_dir) + 1:]

def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)