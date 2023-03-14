from glob import glob
import numpy as np
import Imath
import OpenEXR
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize, ColorJitter
from torch.utils.data import Dataset, random_split, DataLoader
from torch import from_numpy
import torch
from pathlib import Path
import os

ndim = 3
num_workers = 4

def exr_loader(path):
    file = OpenEXR.InputFile(path)

    dw = file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

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

def load(path):
    if not path.endswith("exr"):
        return to_tensor(default_loader(path))
    else:
        return from_numpy(exr_loader(path))

class ImageDataSet(Dataset):
    def __init__(self, items, transforms, global_transform):
        self.items = items
        self.transforms = transforms
        self.global_transform = global_transform

    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, index):
        item = self.items[index]
        item = {name: load(path) for name, path in item.items()}
        
        out = {}
        for name, ten in item.items():
            if self.global_transform != None:
                ten = self.global_transform(ten)

            if self.transforms.__contains__(name):
                ten = self.transforms[name](ten)

            out[name] = ten

        return out


def split_dataset(dataset, valid_perc, test_perc):
    train_perc = 1.0 - test_perc - valid_perc
    return random_split(dataset, [train_perc, valid_perc, test_perc])

def load_dataset(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

def load_datasets(dir, missing_names, transforms, datasets, batch_size, valid_perc, test_perc, global_transform=None):
    items = []

    inst0_name = datasets[0][0][0][0]
    for (instances, count) in datasets[1:]:
        if inst0_name != instances[0][0]:
            print("first instance name must always be the same")
            exit()

    missing_names = {name: f"{dir}/{path}" for name, path in missing_names.items()}

    for (instances, count) in datasets:
        (inst0_name, inst0_subdir, inst0_ext) = instances[0]
        inst0_paths = glob(f"{dir}/{inst0_subdir}/*.{inst0_ext}")

        for inst0_path in inst0_paths[:count]:        
            filename = Path(inst0_path).stem
            item = {inst0_name: inst0_path}

            for (name, subdir, ext) in instances[1:]:
                path = f"{dir}/{subdir}/{filename}.{ext}"
                
                if os.path.exists(path):
                    item[name] = path
                else:
                    print("path is missing.")
                    exit()

            for name, path in missing_names.items():
                if not name in item:
                    if os.path.exists(path):
                        item[name] = path
                    else:
                        print("missing path is missing")
                        exit()

            items.append(item)

    np.random.shuffle(items)

    ds = ImageDataSet(items, transforms, global_transform)
    train_ds, valid_ds, test_ds = split_dataset(ds, valid_perc, test_perc)
    train_ds, valid_ds, test_ds = load_dataset(train_ds, batch_size), load_dataset(valid_ds, batch_size), load_dataset(test_ds, batch_size)

    return train_ds, valid_ds, test_ds


resize_transform = Resize((64, 64))
jitter_transform = ColorJitter(brightness=0.1, contrast=0.05, saturation=0.3, hue=0.1)

def load_pre_dataset(batch_size):
    return load_datasets("/media/shared/Projekte/DocumentScanner/datasets", {"uv": "blank_uv.exr"}, {"img": jitter_transform}, [
        ([("img", "Doc3d_64x64/img/1", "png"), ("uv", "Doc3d_64x64/lines/1", "png")], 5000),
        ([("img", "Doc3d_64x64/img/2", "png"), ("uv", "Doc3d_64x64/lines/2", "png")], 5000),
        ([("img", "Doc3d_64x64/img/3", "png"), ("uv", "Doc3d_64x64/lines/3", "png")], 5000),
        ([("img", "MitIndoor_64x64", "jpg")], 1500)
    ], batch_size=batch_size, valid_perc=0.1, test_perc=0.1, global_transform=resize_transform)

def load_bm_dataset(batch_size):
    return load_datasets("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [], {"img_masked": jitter_transform}, [
        ([("img_masked", "img_masked/1", "png"), ("bm", "bm/1exr", "exr"), ("uv", "uv/1", "exr")], 5000),
        ([("img_masked", "img_masked/2", "png"), ("bm", "bm/2exr", "exr"), ("uv", "uv/2", "exr")], 5000),
        ([("img_masked", "img_masked/3", "png"), ("bm", "bm/3exr", "exr"), ("uv", "uv/3", "exr")], 5000)
    ], batch_size=batch_size, valid_perc=0.1, test_perc=0.1, global_transform=resize_transform)