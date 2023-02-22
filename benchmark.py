#!/usr/bin/python3
from model import load_model, binarize
from data import prepare_datasets
from torchmetrics import CharErrorRate, ExtendedEditDistance 
import torch
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import pytesseract

bmap_padding = 1e-2
num_examples = 3
device = 'cuda'


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


ds, _, _ = prepare_datasets([
    ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", "img/1", "bm/1exr", "png", "exr", 20000)
], valid_perc=0.1, test_perc=0.1, batch_size=num_examples)


model = load_model("model.pth")
model.eval()

(img, label) = next(iter(ds))

# img = img.detach().cpu().numpy()
# label = label.detach().cpu().numpy()

# img = np.transpose(img[0], [1, 2, 0])
# label = np.transpose(label[0], [1, 2, 0])[:,:,0:2]

# print(img.shape, img.dtype, label.shape, label.dtype)
# print(label.min(), label.max())

# out = cv2.remap(img, label, None, interpolation=cv2.INTER_AREA)
# cv2.imwrite("test.png", out * 255)

# exit()

mask_label = label[:, 2, :, :].unsqueeze(axis=1)
mask_label = binarize(mask_label)
img_mask_label = torch.cat([img, mask_label], axis=1) # mask_label * img

mask_pred = model.mask_model(img)
uv_pred = model.uv_model(img_mask_label)
full_pred = model(img)

img = img.detach().cpu().numpy()
label = label.detach().cpu().numpy()

mask_pred = mask_pred.detach().cpu().numpy()
uv_pred = uv_pred.detach().cpu().numpy()
full_pred = full_pred.detach().cpu().numpy()


bmaps = [bmap_from_fmap(label[e,0:2,:,:], label[e,2,:,:]) for e in range(num_examples)]
bmaps = [sample_from_bmap(np.transpose(img[e,:,:,:], [1, 2, 0]), bmap) for e, bmap in enumerate(bmaps)]

img = np.transpose(img, [0, 2, 3, 1])
label = np.transpose(label, [0, 2, 3, 1])

mask_pred = np.transpose(mask_pred, [0, 2, 3, 1])
uv_pred = np.transpose(uv_pred, [0, 2, 3, 1])
full_pred = np.transpose(full_pred, [0, 2, 3, 1])

plt.figure(figsize=(25, 25))

title = ['image', 'uv + mask label', 'uv + mask prediction', 'backwards mapping']

i = 0
for e in range(num_examples):
    list = [img[e,:,:,:], label[e,:,:,:], np.concatenate([uv_pred[e,:,:,:], mask_pred[e,:,:,:]], axis=-1), bmaps[e]]
    for t, arr in enumerate(list):
        plt.subplot(num_examples, len(title), i + 1)
        plt.title(title[t])
        plt.imshow(arr)

        plt.axis('off')
        i += 1

plt.show()


# cer = 0.0
# ed = 0.0
# counter = 0

# char_error_rate = CharErrorRate()
# edit_distance = ExtendedEditDistance()

# text_label = str(pytesseract.image_to_string(unwrapped_label))
# text_pred = str(pytesseract.image_to_string(unwrapped_pred))

# cer += char_error_rate(text_pred, text_label).item()
# ed += edit_distance(text_pred, text_label).item()
# counter += 1