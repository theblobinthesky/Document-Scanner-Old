#!/usr/bin/python3
from data import load_pre_dataset, load_bm_dataset
from torchvision.transforms import Resize
import torch
import numpy as np
import matplotlib.pyplot as plt

bmap_padding = 1e-2
num_examples = 8

dpi = 50

def sample_from_bmap(img, bmap):
    bmap = bmap.clone().permute(0, 2, 3, 1)
    bmap = bmap * 2 - 1

    out = torch.nn.functional.grid_sample(img, bmap, align_corners=False)                
    return out


transform = Resize((128, 128))

def filter(list):
    return [list[0], list[len(list) // 2], list[-1]]

def cpu(ten):
    return np.transpose(ten.detach().cpu().numpy(), [0, 2, 3, 1]).squeeze(axis=0)

def pad_if_necessary(ten):
    channels = ten.shape[-1]
    if channels == 3 or channels == 1:
        return ten
    elif channels == 2:
        return np.pad(ten, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    elif channels == 4:
        return ten[:, :, :3]
    else:
        print("error invalid channel number")
        exit()


def benchmark_plt_pre(model):
    _, _, ds, = load_pre_dataset(1)

    iterator = iter(ds)
    xs, ys, preds = [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to("cuda")

        x, y = model.input_and_label_from_dict(dict)

        pred = model(x)

        x, y, pred = cpu(x), cpu(y), cpu(pred)
        x, y, pred = pad_if_necessary(x), pad_if_necessary(y), pad_if_necessary(pred)

        xs.append(x)
        ys.append(y)
        preds.append(pred)

    title = ["image", "mask label", "mask pred"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        list = [xs[e], preds[e], ys[e]]
        
        for t, arr in enumerate(list):
            plt.subplot(num_examples, len(title), i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig


def benchmark_plt_bm(model, ds):
    _, _, ds = load_bm_dataset(1)

    iterator = iter(ds)
    xs, ys, preds, sampleds = [], [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to("cuda")

        x, y = model.input_and_label_from_dict(dict)

        pred = model.forward_all(x)
        
        sampled = [sample_from_bmap(x, p) for p in pred]

        x, y = cpu(x), cpu(y)
        pred = [cpu(p) for p in pred]
        sampled = [cpu(s) for s in sampled]

        x, y = pad_if_necessary(x), pad_if_necessary(y)
        pred = [pad_if_necessary(p) for p in pred]
        sampled = [pad_if_necessary(s) for s in sampled]

        xs.append(x)
        ys.append(y)
        preds.append(pred)
        sampleds.append(sampled)


    sample_titles = [f"sampled {i + 1}" for i in range(len(sampleds[0]))]
    title = ["image masked", "uv label", *filter(sample_titles)]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        list = [xs[e], ys[e], *filter(sampleds[e])]
        
        for t, arr in enumerate(list):
            plt.subplot(num_examples, len(title), i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig


from data import load_bm_dataset
import bm_model

if __name__ == "__main__":
    ds = load_bm_dataset()
  
    model = bm_model.BMModel(True)
    model.load_state_dict(torch.load("models/bm_progressive_baseline.pth"))
    model = model.cuda()

    fig = benchmark_plt_bm(model, ds)
    fig.savefig("docs/fig.png", dpi=90)