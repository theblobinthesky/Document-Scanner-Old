#!/usr/bin/python3
from data import load_datasets
from torchvision.transforms import Resize
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

bmap_padding = 1e-2
num_examples = 6

dpi = 50

def sample_from_bmap(img, bmap):
    bmap = bmap.clone().permute(0, 2, 3, 1)
    bmap = bmap * 2 - 1

    out = torch.nn.functional.grid_sample(img, bmap, align_corners=False)                
    return out


transform = Resize((128, 128))

def filter(list):
    return [list[0], list[len(list) // 2], list[-1]]


def benchmark_plt(model, ds, is_rnn):
    _, _, ds = load_datasets(*ds, 1)

    iterator = iter(ds)
    xs, ys, preds, sampleds = [], [], [], []
    for i in range(num_examples):
        dict = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to("cuda")

        x, y = model.input_and_label_from_dict(dict)

        if is_rnn:
            pred = model.forward_all(x)
        else:
            pred = [model(x)]
        
        sampled = [sample_from_bmap(x, p) for p in pred]

        def cpu(ten):
            return np.transpose(ten.detach().cpu().numpy(), [0, 2, 3, 1]).squeeze(axis=0)

        x, y = cpu(x), cpu(y)
        pred = [cpu(p) for p in pred]
        sampled = [cpu(s) for s in sampled]

        def pad_if_necessary(ten):
            channels = ten.shape[-1]
            if channels == 3 or channels == 1:
                return ten
            elif channels == 2:
                return np.pad(ten, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
            else:
                print("error invalid channel number")
                exit()

        x, y = pad_if_necessary(x), pad_if_necessary(y)
        pred = [pad_if_necessary(p) for p in pred]
        sampled = [pad_if_necessary(s) for s in sampled]

        xs.append(x)
        ys.append(y)
        preds.append(pred)
        sampleds.append(sampled)


    sample_titles = [f"sampled {i + 1}" for i in range(len(sampleds[0]))]
    title = ["x", "y", *filter(sample_titles)]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in tqdm(range(num_examples), desc="Plotting benchmark"):
        list = [xs[e], ys[e], *filter(sampleds[e])]
        
        for t, arr in enumerate(list):
            plt.subplot(num_examples, len(title), i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig


from data import prepare_bm_dataset
import bm_model

if __name__ == "__main__":
    ds = prepare_bm_dataset()
  
    model = bm_model.BMModel(True)
    model.load_state_dict(torch.load("models/test.pth"))
    model = model.cuda()

    fig = benchmark_plt(model, ds, True)
    fig.savefig("docs/fig.png", dpi=50)