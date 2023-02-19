from glob import glob
import numpy as np
import OpenEXR
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from pathlib import Path
import os

class ImageDataSet(Dataset):
    def __init__(self, pairs, loader, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)
        
    def load(self, path):
        if path.endswith("exr"):
            return None
        else:
            return default_loader(path)

    def __getitem__(self, index):
        (inp, label) = self.pairs[index]

        x = load(inp), load(label)

        if self.transform != None:
            x = self.transform(x)

        return x

def prepare_datasets(sets):
    pairs = []

    for (dir, inp_subdir, label_subdir, inp_ext, label_ext, count) in sets:
        inp_paths = glob(f"{dir}/{inp_subdir}/*.{inp_ext}")

        for inp in inp_paths[:count]:
            name = Path(inp).stem
            label = f"{dir}/{label_subdir}/{name}.{label_ext}"

            print(inp, label)
            if not os.path.exists(label):
                print("label is missing.")
                exit()

            pairs.append((inp, label))

    np.random.shuffle(pairs)

    ds = ImageDataSet(pairs, None, None)
    return iter(ds)