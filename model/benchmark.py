#!/usr/bin/python3
from model import binarize_threshold
from data import load_pre_dataset, load_bm_dataset
from torchvision.transforms import Resize
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

num_examples = 8
dpi = 50
circle_radius = 2
circle_colors = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
]

def sample_from_bmap(img, bmap):
    bmap = bmap.clone().permute(0, 2, 3, 1)
    bmap = bmap * 2 - 1

    out = torch.nn.functional.grid_sample(img, bmap, align_corners=False)                
    return out


transform = Resize((128, 128))

def filter(list):
    return [list[0], list[len(list) // 2], list[-1]]

def cpu(ten, only_detach=False):
    if only_detach:
        return ten.detach().cpu().numpy()
    else:
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

def apply_flatten_to_mask(mask, flatten):
    mask = mask.repeat(3, axis=-1)

    exists = flatten[0, 0]
    if exists < binarize_threshold:
        return np.zeros(mask.shape, dtype=np.float32)

    def get_pt(start):
        pt = (flatten[0, start:start+2] * 64.0).astype("int32")
        return (pt[1], pt[0])
    
    tl, tr, br, bl = get_pt(1), get_pt(3), get_pt(5), get_pt(7)

    cv2.circle(mask, tl, circle_radius, circle_colors[0])
    cv2.circle(mask, tr, circle_radius, circle_colors[1])
    cv2.circle(mask, br, circle_radius, circle_colors[2])
    cv2.circle(mask, bl, circle_radius, circle_colors[3])

    return mask

def benchmark_plt_pre(model):
    _, _, ds, = load_pre_dataset(1)

    iterator = iter(ds)
    xs, ys, preds = [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to("cuda")

        x, (mask_label, flatten_label) = model.input_and_label_from_dict(dict)

        mask_pred, flatten_pred = model(x)

        x, mask_label, flatten_label, mask_pred, flatten_pred = cpu(x), cpu(mask_label), cpu(flatten_label, True), cpu(mask_pred), cpu(flatten_pred, True)
        
        xs.append(x)
        ys.append((flatten_label, mask_label))
        preds.append((flatten_pred, mask_pred))

    title = ["image", "mask pred", "mask label"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        mask_label, flatten_label = ys[e]
        mask_pred, flatten_pred = preds[e]

        mask_label = apply_flatten_to_mask(flatten_label, mask_label)
        mask_pred = apply_flatten_to_mask(flatten_pred, mask_pred)

        list = [xs[e], mask_pred, mask_label]
        
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