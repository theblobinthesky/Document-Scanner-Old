from glob import glob
import numpy as np
import Imath
import OpenEXR
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.folder import default_loader
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
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, index):
        (inputs, label) = self.pairs[index]

        xs = [load(inp) for inp in inputs]
        y = load(label)

        if self.transform != None:
            xs = [self.transform(x) for x in xs]
            y = self.transform(y)

        x = np.concatenate(xs, axis=0)

        return x, y

def split_dataset(dataset, valid_perc, test_perc):
    train_perc = 1.0 - test_perc - valid_perc
    return random_split(dataset, [train_perc, valid_perc, test_perc])

def load_dataset(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

def prepare_datasets(sets, valid_perc, test_perc, batch_size, transform=None):
    pairs = []

    for (dir, inp_sets, label_subdir, label_ext, count) in sets:
        inp_subdir, inp_ext = inp_sets[0]
        inp_paths = glob(f"{dir}/{inp_subdir}/*.{inp_ext}")

        for inp in inp_paths[:count]:
            name = Path(inp).stem
            label = f"{dir}/{label_subdir}/{name}.{label_ext}"
            if not os.path.exists(label):
                print("label is missing.")
                exit()

            inputs = [inp]
            
            for (inp_subdir, inp_ext) in inp_sets[1:]:   
                input = f"{dir}/{inp_subdir}/{name}.{inp_ext}"
                inputs.append(input)

                if not os.path.exists(input):
                    print("input is missing.")
                    exit()

             
            pairs.append((inputs, label))


    np.random.shuffle(pairs)

    ds = ImageDataSet(pairs, transform)
    train_ds, valid_ds, test_ds = split_dataset(ds, valid_perc, test_perc)

    return load_dataset(train_ds, batch_size), \
           load_dataset(valid_ds, batch_size), \
           load_dataset(test_ds, batch_size)