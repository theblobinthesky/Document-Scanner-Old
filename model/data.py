from glob import glob
import numpy as np
import Imath
import OpenEXR
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize, ColorJitter
from torch.utils.data import Dataset, random_split, DataLoader
from torch import from_numpy
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
    def __init__(self, items, names, transforms, global_transform):
        self.items = items
        self.names = names
        self.transforms = transforms
        self.global_transform = global_transform

    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, index):
        item = self.items[index]
        item = [load(path) for path in item]

        out = {}
        for i, name in enumerate(self.names):
            if self.global_transform != None:
                item[i] = self.global_transform(item[i])

            if self.transforms.__contains__(name):
                item[i] = self.transforms[name](item[i])

            out[name] = item[i]

        return out


def split_dataset(dataset, valid_perc, test_perc):
    train_perc = 1.0 - test_perc - valid_perc
    return random_split(dataset, [train_perc, valid_perc, test_perc])

def load_dataset(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

def prepare_datasets(dir, exts, transforms, datasets, valid_perc, test_perc, global_transform=None):
    for (instances, count) in datasets:
        for (name, subdir) in instances:
            if not exts.__contains__(name):
                print("name is unknown.")
                exit()
    
    for name in exts:
        for (instances, count) in datasets:
            found = False

            for (inst_name, subdir) in instances:
                if name == inst_name:
                    found = True
                    break

            if not found:
                print("instance name is missing")
                exit()

    items = []
    names = exts.keys()

    for (instances, count) in datasets:
        (name, subdir) = instances[0]
        paths = glob(f"{dir}/{subdir}/*.{exts[name]}")

        for path in paths[:count]:        
            filename = Path(path).stem
            item = [path]

            for (name, subdir) in instances[1:]:
                path = f"{dir}/{subdir}/{filename}.{exts[name]}"
                item.append(path)

                if not os.path.exists(path):
                    print("path is missing.")
                    exit()

            items.append(item)

    np.random.shuffle(items)

    ds = ImageDataSet(items, names, transforms, global_transform)
    train_ds, valid_ds, test_ds = split_dataset(ds, valid_perc, test_perc)

    return train_ds, valid_ds, test_ds


def load_datasets(train_ds, valid_ds, test_ds, batch_size):
    return load_dataset(train_ds, batch_size), \
           load_dataset(valid_ds, batch_size), \
           load_dataset(test_ds, batch_size)


resize_transform = Resize((64, 64))
jitter_transform = ColorJitter(brightness=0.1, contrast=0.05, saturation=0.3, hue=0.1)

def prepare_pre_dataset():
    return prepare_datasets("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", {"img": "png", "uv": "exr"}, {}, [
        ([("img", "img/1"), ("uv", "uv/1")], 5000),
        ([("img", "img/2"), ("uv", "uv/2")], 5000),
        ([("img", "img/3"), ("uv", "uv/3")], 5000)
    ], valid_perc=0.1, test_perc=0.1, global_transform=resize_transform)

def prepare_bm_dataset():
    return prepare_datasets("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", {"img_masked": "png", "bm": "exr", "uv": "exr"}, {"img_masked": jitter_transform}, [
        ([("img_masked", "img_masked/1"), ("bm", "bm/1exr"), ("uv", "uv/1")], 5000),
        ([("img_masked", "img_masked/2"), ("bm", "bm/2exr"), ("uv", "uv/2")], 5000),
        ([("img_masked", "img_masked/3"), ("bm", "bm/3exr"), ("uv", "uv/3")], 5000)
    ], valid_perc=0.1, test_perc=0.1, global_transform=resize_transform)