#!/usr/bin/python3
from model import load_model, binarize
from data import prepare_datasets
from torchvision.transforms import Resize
from torchmetrics import CharErrorRate, ExtendedEditDistance 
import torch
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import io

bmap_padding = 1e-2
num_examples = 8
device = torch.device('cpu')

dpi = 50

def bmap_from_fmap(fmap, mask):
    indices = np.indices(fmap.shape[1:]).transpose([2, 1, 0]).reshape((-1, 2))
    sample_at = (indices / indices.max() + bmap_padding) * (1.0 - 2.0 * bmap_padding)
    # todo: adjust max(dim=(..)) and bmap_padding for other aspect-ratio if necessary in the future 
    fmap = fmap.transpose([2, 1, 0]).reshape((-1, 2))

    u_l, i_l = [], []

    for i in range(0, mask.shape[1] * mask.shape[0], 1):
        ind = indices[i]
        
        if mask[ind[0], ind[1]] == 1.0:
            u_l.append(fmap[i].tolist())
            i_l.append(indices[i].tolist())

    u_l = np.array(u_l, np.float32)
    i_l = np.array(i_l, np.float32)
    i_l[:, 0] /= float(mask.shape[1] - 1)
    i_l[:, 1] /= float(mask.shape[0] - 1)

    bmap = scipy.interpolate.LinearNDInterpolator(u_l, i_l)
    bmap = bmap(sample_at).reshape((mask.shape[0], mask.shape[1], 2))
    
    bmap = bmap.astype("float32")
    return bmap


def sample_from_bmap(img, bmap):
    bmap = 1.0 - bmap
    bmap[:, :, 0] *= (img.shape[1] - 1)
    bmap[:, :, 1] *= (img.shape[0] - 1)
    bmap = cv2.resize(bmap, (128, 128))

    out = cv2.remap(img, bmap, None, interpolation=cv2.INTER_AREA)
    return out

# bmaps = [bmap_from_fmap(label[e,0:2,:,:], label[e,2,:,:]) for e in range(num_examples)]
# bmaps = [sample_from_bmap(np.transpose(img[e,:,:,:], [1, 2, 0]), bmap) for e, bmap in enumerate(bmaps)]

def benchmark_eval(model):
    transform = Resize((128, 128))

    # ds, _, _ = prepare_datasets([
    #     ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("img/1", "png")], "wc/1", "exr", 20000)
    # ], valid_perc=0.1, test_perc=0.1, batch_size=num_examples, transform=transform)

    # x, y = next(iter(ds))


    ds, _, _ = prepare_datasets("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", 
        {"img": "png", "lines": "png"}, [
        ([("img", "img/1"), ("lines", "lines/1")], 100)
    ], valid_perc=0.1, test_perc=0.1, batch_size=num_examples, transform=transform)

    dict = next(iter(ds))

    # x, bm_label = next(iter(ds))
    # x[:, 0:3] *= x[:, 3].unsqueeze(axis=1)
    # pre_label = x[:, :4]
    # wc_label = x[:, 6:9]
    # bm_label = bm_label[:, :2] / 448.0
    # img_label = x[:, 9:12]

    img_label = dict["img"]
    pre_label = dict["lines"][:, [0, 2]]

    pre_pred = model(img_label.to('cuda'))
    # wc_pred = model.wc_model(pre_label)
    # bm_pred = model.bm_model(wc_label)


    def cpu(ten):
        return np.transpose(ten.detach().cpu().numpy(), [0, 2, 3, 1])

    img_label = cpu(img_label)
    pre_pred, pre_label = cpu(pre_pred), cpu(pre_label)
    # wc_label, wc_pred = cpu(wc_label), cpu(wc_pred)
    # bm_label, bm_pred = cpu(bm_label), cpu(bm_pred)


    # def unwrap(e, ten):
    #     return cv2.remap(pre_label[e, :, :, :3], ten * 128.0, None, interpolation=cv2.INTER_LINEAR)

    # unwarped_label = [unwrap(e, bm_label[e]) for e in range(num_examples)]
    # unwarped_pred = [unwrap(e, bm_pred[e]) for e in range(num_examples)]

    def pad(ten):
        return np.pad(ten, ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

    pre_pred, pre_label = pad(pre_pred), pad(pre_label)

    # title = ['pre pred', 'wc label', 'wc prediction', 'bm label', 'bm prediction', 'unwarped label', 'unwarped prediction']
    title = ["img", "pre pred", "pre label"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        list = [img_label[e], pre_pred[e], pre_label[e]]
        # list = [pre_pred[e,:,:,:3], wc_label[e], wc_pred[e], bm_label[e], bm_pred[e], unwarped_label[e], unwarped_pred[e]]
        for t, arr in enumerate(list):
            plt.subplot(num_examples // 2, len(title) * 2, i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig