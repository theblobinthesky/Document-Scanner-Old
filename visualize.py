#!/usr/bin/python3
from model import load_model
from data import prepare_datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2

bmap_padding = 1e-3

def bmap_from_fmap(fmap, mask):
    indices = np.indices(fmap.shape[1:]).transpose([2, 1, 0]).reshape((-1, 2))
    sample_at = (indices / indices.max() + bmap_padding) * (1.0 - 2.0 * bmap_padding)
    # todo: adjust max(dim=(..)) and bmap_padding for other aspect-ratio if necessary in the future 
    fmap = fmap.reshape((-1, 2))

    u_l, i_l = [], []

    for i in range(0, mask.shape[1] * mask.shape[0], 1):
        ind = indices[i]
        
        if mask[ind[1], ind[0]] == 1.0:
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

    print(img.shape, bmap.shape)
    out = cv2.remap(img, bmap, None, interpolation=cv2.INTER_AREA)
    print(out.shape)
    return out


num_examples = 3
device = 'cuda'

ds, _, _ = prepare_datasets([
    ("/media/shared/Projekte/Scanner/datasets/Doc3d", "img", "uv_exr", "png", "exr", 100)
], valid_perc=0.1, test_perc=0.1, batch_size=num_examples, device=device)

model = load_model("model.pth")
model.eval()

(img, uv_label) = next(iter(ds))
uv_pred = model(img)

img = img.detach().cpu().numpy()
uv_label = uv_label.detach().cpu().numpy()
uv_pred = uv_pred.detach().cpu().numpy()

bmaps = [bmap_from_fmap(uv_label[e,0:2,:,:], uv_label[e,2,:,:]) for e in range(num_examples)]
bmaps = [sample_from_bmap(np.transpose(img[e,:,:,:], [1, 2, 0]), bmap) for e, bmap in enumerate(bmaps)]

img = np.transpose(img, [0, 2, 3, 1])
uv_label = np.transpose(uv_label, [0, 2, 3, 1])
uv_pred = np.transpose(uv_pred, [0, 2, 3, 1])

plt.figure(figsize=(25, 25))

title = ['image', 'uv label', 'uv prediction', 'backward map sampled']

i = 0
for e in range(num_examples):
    list = [img[e,:,:,:], uv_label[e,:,:,2], uv_pred[e,:,:,:], bmaps[e]]
    for t, arr in enumerate(list):
        plt.subplot(num_examples, len(title), i + 1)
        plt.title(title[t])
        plt.imshow(arr)

        plt.axis('off')
        i += 1

plt.show()