#!/usr/bin/python3
import os
import subprocess
import sys
from tqdm import tqdm

# Based on:
# https://github.com/cdcseacave/openMVS/blob/master/MvgMvsPipeline.py

OPENMVG_BIN = "/usr/local/bin"
OPENMVS_BIN = OPENMVG_BIN

CAMERA_SENSOR_DB_FILE = "sensor_width_camera_database.txt"
CAMERA_SENSOR_DB_DIRECTORY = "/home/erik/Downloads/ReleaseV2.0.Rainbow-Trout.WindowsBinaries_VS2019/exif/sensor_width_database/"

class AStep:
    """ Represents a process step to be run """
    def __init__(self, info, cmd, opt):
        self.info = info
        self.cmd = cmd
        self.opt = opt


input_dir = "/home/erik/Downloads/ReleaseV2.0.Rainbow-Trout.WindowsBinaries_VS2019/ImageDataset_SceauxCastle/images"
output_dir = "/home/erik/Downloads/ReleaseV2.0.Rainbow-Trout.WindowsBinaries_VS2019/ImageDataset_SceauxCastle/output"
reconstruction_dir = "/home/erik/Downloads/ReleaseV2.0.Rainbow-Trout.WindowsBinaries_VS2019/ImageDataset_SceauxCastle/reconstruction"

if not os.path.exists(input_dir):
    sys.exit(f"input path not found: {input_dir}")


def replace_opt(self, idx, str_exist, str_new):
    s = self.steps_data[idx]
    o2 = []
    for o in s[2]:
        co = o.replace(str_exist, str_new)
        o2.append(co)
    s[2] = o2

def make_steps():
    return [
        ["Intrinsics analysis",
         os.path.join(OPENMVG_BIN, "openMVG_main_SfMInit_ImageListing"),
         ["-i", input_dir, "-o", matches_dir, "-d", camera_file_params]],
        ["Compute features",
         os.path.join(OPENMVG_BIN, "openMVG_main_ComputeFeatures"),
         ["-i", f"{matches_dir}/sfm_data.json", "-o", matches_dir, "-m", "SIFT"]],
        ["Compute pairs",
         os.path.join(OPENMVG_BIN, "openMVG_main_PairGenerator"),
         ["-i", f"{matches_dir}/sfm_data.json", "-o", f"{matches_dir}/pairs.bin"]],
        ["Compute matches",
         os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
         ["-i", f"{matches_dir}/sfm_data.json", "-p", f"{matches_dir}/pairs.bin", "-o", f"{matches_dir}/matches.putative.bin", "-n", "AUTO"]],
        ["Filter matches",
         os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
         ["-i", f"{matches_dir}/sfm_data.json", "-m", f"{matches_dir}/matches.putative.bin", "-o", f"{matches_dir}/matches.f.bin"]],
        ["Incremental reconstruction",
         os.path.join(OPENMVG_BIN, "openMVG_main_SfM"),
         ["-i", f"{matches_dir}/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir, "-s", "INCREMENTAL"]],
        ["Export to openMVS",
         os.path.join(OPENMVG_BIN, "openMVG_main_openMVG2openMVS"),
         ["-i", f"{reconstruction_dir}/sfm_data.bin", "-o", f"{mvs_dir}/scene.mvs", "-d", f"{mvs_dir}/images"]],
        ["Densify point cloud",
         os.path.join(OPENMVS_BIN, "MVSDensifyPointCloud"),
         ["scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "1", "--number-views", "8", "-w", f"\"{mvs_dir}\""]],
        ["Reconstruct the mesh",
         os.path.join(OPENMVS_BIN, "MVSReconstructMesh"),
         ["scene_dense.mvs", "-w", f"\"{mvs_dir}\""]]
        # ["Refine the mesh",
        #  os.path.join(OPENMVS_BIN, "MVSRefineMesh"),
        #  ["scene_dense_mesh.mvs", "--scales", "1", "--gradient-step", "25.05", "-w", f"\"{mvs_dir}\""]]
        ]


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


reconstruction_dir, matches_dir = os.path.join(output_dir, "sfm"), os.path.join(reconstruction_dir, "matches") 
mvs_dir = os.path.join(output_dir, "mvs")
camera_file_params = os.path.join(CAMERA_SENSOR_DB_DIRECTORY, CAMERA_SENSOR_DB_FILE)

make_dirs([output_dir, reconstruction_dir, matches_dir, mvs_dir])

steps = make_steps()

# RefineMesh step is not run, use ReconstructMesh output
# steps.replace_opt(15, "scene_dense_mesh_refine.mvs", "scene_dense_mesh.mvs")

def solve_in_3d():
    pbar = tqdm(total=len(steps))

    for [name, path, args] in steps:
        pbar.set_description(f"3d solver ({name})")

        try:
            cmdline = [path] + args
            pipes = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            std_out, std_err = pipes.communicate()
            
            if pipes.returncode != 0:
                err_msg = "%s. Code: %s" % (std_err.strip(), pipes.returncode)
                print(err_msg)
                exit()

        except KeyboardInterrupt:
            sys.exit('\r\nProcess canceled by user, all files remains')

        pbar.update()

    pbar.close()

solve_in_3d()