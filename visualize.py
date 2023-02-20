#!/usr/bin/python3
from model import load_model
from data import prepare_datasets
import matplotlib.pyplot as plt
import numpy as np

num_examples = 5
device = 'cuda'

ds, _, _ = prepare_datasets([
    ("/media/shared/Projekte/Scanner/datasets/Doc3d", "img", "uv_exr", "png", "exr", 100)
], valid_perc=0.1, test_perc=0.1, batch_size=num_examples, device=device)

model = load_model("model 2.pth")
model.eval()

(img, uv_label) = next(iter(ds))
uv_pred = model(img)
uv_pred = uv_pred.detach().cpu()

plt.figure(figsize=(25, 25))

title = ['image', 'uv label', 'uv prediction']

i = 0
for e in range(num_examples):
    list = [img[e,:,:,:], uv_label[e,:,:,:], uv_pred[e,:,:,:]]
    for t, arr in enumerate(list):
        arr = np.transpose(arr, [1, 2, 0])
        plt.subplot(num_examples, len(title), i + 1)
        plt.title(title[t])
        plt.imshow(arr)

        plt.axis('off')
        i += 1

plt.show()

