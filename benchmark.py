#!/usr/bin/python3
from model import load_model, binarize
from data import load_datasets
import data
from torchvision.transforms import Resize
import torch
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt

bmap_padding = 1e-2
num_examples = 16

dpi = 50

def sample_from_bmap(img, bmap):
    bmap = bmap.clone().permute(0, 2, 3, 1)
    bmap = bmap * 2 - 1

    out = torch.nn.functional.grid_sample(img, bmap, align_corners=False)                
    return out


transform = Resize((128, 128))

def benchmark_plt(model, ds):
    _, _, ds = load_datasets(*ds, 1)

    iterator = iter(ds)
    xs, ys, preds, sampleds = [], [], [], []
    for i in range(num_examples):
        dict = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to("cuda")

        x = model.input_from_dict(dict)
        pred = model(x)
        sampled = sample_from_bmap(x, pred)

        def cpu(ten):
            return np.transpose(ten.detach().cpu().numpy(), [0, 2, 3, 1]).squeeze(axis=0)

        x, y, pred, sampled = cpu(x), cpu(y), cpu(pred), cpu(sampled)

        def pad_if_necessary(ten):
            channels = ten.shape[-1]
            if channels == 3 or channels == 1:
                return ten
            elif channels == 2:
                return np.pad(ten, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
            else:
                print("error invalid channel number")
                exit()

        x, y, pred, sampled = pad_if_necessary(x), pad_if_necessary(y), pad_if_necessary(pred), pad_if_necessary(sampled)

        xs.append(x)
        ys.append(y)
        preds.append(pred)
        sampleds.append(sampled)


    title = ["x", "y", "pred", "sampled"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        list = [xs[e], ys[e], preds[e], sampleds[e]]
        for t, arr in enumerate(list):
            plt.subplot(num_examples // 2, len(title) * 2, i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig