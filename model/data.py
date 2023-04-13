from glob import glob
import numpy as np
import Imath
import OpenEXR
from torchvision.transforms.functional import to_tensor
import cv2
from torchvision.transforms import Resize, ColorJitter, Compose
from torch.utils.data import Dataset, random_split, DataLoader
from torch import from_numpy
import torch
from pathlib import Path
import os

num_workers = 4

def exr_loader(path):
    file = OpenEXR.InputFile(path)

    header = file.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    ndim = len(header["channels"])

    def load_channel(name):
        C = file.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.frombuffer(C, np.float32)
        C = np.reshape(C, size)
        return C

    if ndim == 1:
        return load_channel('R')
    elif ndim == 3:
        channels = [load_channel(c)[np.newaxis, :] for c in ['R', 'G', 'B']]
        return np.concatenate(channels, axis=0)
    else:
        print("incorrect number of channels.")
        exit()


def npy_loader(name, path):
    with open(path, "rb") as f:
        npy = np.load(f, allow_pickle=True)

    # if isinstance(npy.item(), dict):
    #     return { f"{name}/{npy_name}": key for npy_name, key in npy.item().items() }
    # else:
    return { name: from_numpy(npy) }


def load(name, path):
    if path.endswith("exr"):
        return { name: from_numpy(exr_loader(path)) }
    elif path.endswith("npy"):
        return npy_loader(name, path)
    else:
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
        if len(img.shape) == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return { name: to_tensor(img) } # todo: investigate performance of this instead of default_laoder


class ImageDataSet(Dataset):
    def __init__(self, items, inst0_name, dir_len, weight_metrics, transforms, global_transform):
        self.items = items
        self.inst0_name = inst0_name
        self.dir_len = dir_len
        self.weight_metrics = weight_metrics
        self.transforms = transforms
        self.global_transform = global_transform

    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, index):
        item_with_path = self.items[index]
        key = item_with_path[self.inst0_name][self.dir_len + 1:]
        weight_metrics = {name: torch.tensor(weight_metric[key], dtype=torch.float32) for name, weight_metric in self.weight_metrics.items()}
        
        item = {}
        for name, path in item_with_path.items():
            item.update(load(name, path))
            
        out = {}
        for name, ten in item.items():
            if self.global_transform != None:
                ten = self.global_transform(ten)

            if self.transforms.__contains__(name):
                ten = self.transforms[name](ten)

            out[name] = ten

        return out, weight_metrics


def split_dataset(dataset, valid_perc, test_perc):
    train_perc = 1.0 - test_perc - valid_perc
    return random_split(dataset, [train_perc, valid_perc, test_perc])

def load_dataset(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

def load_datasets(dir, missing_names, transforms, datasets, weight_metrics, batch_size, valid_perc, test_perc, global_transform=None):
    items = []

    inst0_name = datasets[0][0][0]
    for instances in datasets[1:]:
        if inst0_name != instances[0][0]:
            print("first instance name must always be the same")
            exit()

    missing_names = {name: f"{dir}/{path}" for name, path in missing_names.items()}

    for instances in datasets:
        (inst0_name, inst0_subdir, inst0_ext) = instances[0]
        inst0_paths = glob(f"{dir}/{inst0_subdir}/*.{inst0_ext}")

        for inst0_path in inst0_paths:        
            filename = Path(inst0_path).stem
            item = {inst0_name: inst0_path}

            for (name, subdir, ext) in instances[1:]:
                path = f"{dir}/{subdir}/{filename}.{ext}"
                
                if os.path.exists(path):
                    item[name] = path
                else:
                    print(f"path '{path}' is missing.")
                    exit()

            for name, path in missing_names.items():
                if not name in item:
                    if os.path.exists(path):
                        item[name] = path
                    else:
                        print(f"missing path '{path}' from '{name}' is missing.")
                        exit()

            items.append(item)

    np.random.shuffle(items)

    # load weight_metrics dictionaries
    for name, path in weight_metrics.items():
        weight_metrics[name] = np.load(f"{dir}/{path}", allow_pickle=True).item() 

    ds = ImageDataSet(items, inst0_name, len(dir), weight_metrics, transforms, global_transform)
    train_ds, valid_ds, test_ds = split_dataset(ds, valid_perc, test_perc)
    train_ds, valid_ds, test_ds = load_dataset(train_ds, batch_size), load_dataset(valid_ds, batch_size), load_dataset(test_ds, batch_size)

    return train_ds, valid_ds, test_ds

color_jitter = ColorJitter(brightness=0.1, contrast=0.05, saturation=0.3, hue=0.1)
resize_64 = Resize(64)
color_comp_64 = Compose([resize_64, color_jitter])
resize_32 = Resize(32)

def load_seg_dataset(batch_size):
    return load_datasets("/media/shared/Projekte/DocumentScanner/datasets", {}, {"img": color_comp_64, "mask": resize_64}, [
        [("img", "Combined/img", "png"), ("mask", "Combined/mask", "png")]
    ], {"finetuning": "Combined/finetuning_metric.npy"}, batch_size=batch_size, valid_perc=0.1, test_perc=0.1)

def load_contour_dataset(batch_size):
    return load_datasets("/media/shared/Projekte/DocumentScanner/datasets", {}, {"img": color_comp_64}, [
        [("img", "Combined/img", "png"), ("contour", "Combined/contour", "npy")]
    ], {"finetuning": "Combined/finetuning_metric.npy"}, batch_size=batch_size, valid_perc=0.1, test_perc=0.1)

def load_bm_dataset(batch_size):
    return load_datasets("/media/shared/Projekte/DocumentScanner/datasets", {}, { 
        "img": resize_32, "img_masked": Compose([resize_32, color_jitter]), "bm": resize_32, "uv": resize_32 }, [
        [("img", "Doc3d_64x64/img/1", "png"), ("img_masked", "Doc3d_64x64/img_masked/1", "png"), ("bm", "Doc3d_64x64/bm/1exr", "exr"), ("uv", "Doc3d_64x64/uv/1", "exr")],
        [("img", "Doc3d_64x64/img/2", "png"), ("img_masked", "Doc3d_64x64/img_masked/2", "png"), ("bm", "Doc3d_64x64/bm/2exr", "exr"), ("uv", "Doc3d_64x64/uv/2", "exr")],
        [("img", "Doc3d_64x64/img/3", "png"), ("img_masked", "Doc3d_64x64/img_masked/3", "png"), ("bm", "Doc3d_64x64/bm/3exr", "exr"), ("uv", "Doc3d_64x64/uv/3", "exr")],
        [("img", "Doc3d_64x64/img/4", "png"), ("img_masked", "Doc3d_64x64/img_masked/4", "png"), ("bm", "Doc3d_64x64/bm/4exr", "exr"), ("uv", "Doc3d_64x64/uv/4", "exr")]
    ], {"finetuning": "finetuning_metric.npy"}, batch_size=batch_size, valid_perc=0.1, test_perc=0.1)