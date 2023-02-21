from glob import glob
import numpy as np
import Imath
import OpenEXR
from torchvision.transforms import Resize, Compose
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
        (inp, label) = self.pairs[index]

        x, y = load(inp), load(label)

        if self.transform != None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

def split_dataset(dataset, valid_perc, test_perc):
    train_perc = 1.0 - test_perc - valid_perc
    return random_split(dataset, [train_perc, valid_perc, test_perc])

def load_dataset(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)

def prepare_datasets(sets, valid_perc, test_perc, batch_size, transform=None):
    pairs = []

    for (dir, inp_subdir, label_subdir, inp_ext, label_ext, count) in sets:
        inp_paths = glob(f"{dir}/{inp_subdir}/*.{inp_ext}")

        for inp in inp_paths[:count]:
            name = Path(inp).stem
            label = f"{dir}/{label_subdir}/{name}.{label_ext}"

            if not os.path.exists(label):
                print("label is missing.")
                exit()

            pairs.append((inp, label))

    np.random.shuffle(pairs)


    if transform == None:
        transform = Resize((128, 128))
    else:
        transform = Compose([
            Resize((128, 128)),
            transform
        ])

    
    ds = ImageDataSet(pairs, transform)
    train_ds, valid_ds, test_ds = split_dataset(ds, valid_perc, test_perc)

    return load_dataset(train_ds, batch_size), \
           load_dataset(valid_ds, batch_size), \
           load_dataset(test_ds, batch_size)