#!/usr/bin/python3
from model import binarize_threshold, device
from data import load_seg_dataset, load_contour_dataset, load_bm_dataset
from seg_model import points_per_side_incl_start_corner
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

num_examples = 8
dpi = 50
circle_radius = 2
rect_color = (0.0, 1.0, 0.0)
circle_color = (1.0, 0.0, 0.0)

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

def benchmark_plt_seg(model):
    _, _, ds, = load_seg_dataset(1)

    iterator = iter(ds)
    xs, ys, preds = [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to(device)

        x, mask_label = model.input_and_label_from_dict(dict)

        mask_pred = model(x)

        x, mask_label, mask_pred = cpu(x), cpu(mask_label), cpu(mask_pred)
        
        xs.append(x)
        ys.append(mask_label)
        preds.append(mask_pred)

    title = ["image", "mask pred", "mask label"]

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

def apply_contour_to_img(img, contour):
    w, h = img.shape[:2]
    img = np.ascontiguousarray(img.copy())

    def swap(pt):
        return (pt[1], pt[0])

    def get_pt(i):
        pt = (contour[0, i] * 64.0).astype("int32")
        return pt
        
    contour_pts = [ get_pt(i) for i in range(4 * points_per_side_incl_start_corner) ]

    for pt in contour_pts:
        cv2.circle(img, swap(pt), circle_radius, circle_color)
        
    return img

def benchmark_plt_contour(model):
    _, _, ds, = load_contour_dataset(1)

    iterator = iter(ds)
    xs, ys, preds = [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to(device)

        x, heatmap_label = model.input_and_label_from_dict(dict)
        heatmap_pred = model(x)
        
        contour_label, contour_pred = model.contour_from_heatmap(heatmap_label), model.contour_from_heatmap(heatmap_pred)

        x = cpu(x)
        heatmap_label, heatmap_pred = cpu(heatmap_label), cpu(heatmap_pred)
        contour_label, contour_pred = cpu(contour_label, True), cpu(contour_pred, True)

        xs.append(x)
        ys.append((contour_label, heatmap_label))
        preds.append((contour_pred, heatmap_pred))


    title = ["pred", "label", "heatmap pred", "heatmap label"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        img = xs[e][:, :, :3]
        
        (contour_pred, heatmap_pred) = preds[e]
        (contour_label, heatmap_label) = ys[e]

        heatmap_pred = heatmap_pred.mean(-1)
        heatmap_label = heatmap_label.mean(-1)

        list = [
            apply_contour_to_img(img, contour_pred),
            apply_contour_to_img(img, contour_label),
            heatmap_pred,
            heatmap_label
        ]
        
        for t, arr in enumerate(list):
            plt.subplot(num_examples, len(title), i + 1)
            plt.title(title[t])
            plt.imshow(arr)

            plt.axis('off')
            i += 1

    return fig


def sample_from_bmap(img, bmap):
    bmap = bmap.clone().permute(0, 2, 3, 1)
    bmap = bmap * 2 - 1

    out = torch.nn.functional.grid_sample(img, bmap, align_corners=False)                
    return out

def benchmark_plt_bm(model, ds):
    _, _, ds = load_bm_dataset(1)

    iterator = iter(ds)
    xs, ys, preds, sampleds = [], [], [], []
    for i in range(num_examples):
        dict, _ = next(iterator)
        for key in dict.keys():
            dict[key] = dict[key].to(device)

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

def benchmark_cam_contour_model(model):
    import pygame

    pygame.init()

    # Set up the Pygame window
    screen = pygame.display.set_mode((640, 480))

    # Initialize the video capture device
    cap = cv2.VideoCapture(2)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(frame).to(device=device)
        img = img.permute((2, 0, 1)).unsqueeze(0)

        b, _, h, w = img.shape
        a_padding = torch.full((b, 1, h, w), 1.0, device=device)
        img = torch.cat([img, a_padding], axis=1)
   
        heatmap = model(img)
        contour = model.contour_from_heatmap(heatmap)
        contour = cpu(contour, True)

        frame = apply_contour_to_img(frame, contour)

        surf = pygame.surfarray.make_surface(np.rot90(frame))
        surf = pygame.transform.scale(surf, (width, height))

        # Display the image in the Pygame window
        screen.blit(surf, (0, 0))
        pygame.display.update()

        # Check for Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                quit()


if __name__ == "__main__":
    from data import load_contour_dataset
    from seg_model import ContourModel

    model = ContourModel()
    model.load_state_dict(torch.load("models/heatmap_model.pth"))
    model = model.to(device=device)

    if False:
        benchmark_cam_contour_model(model)
    else:
        fig = benchmark_plt_contour(model)
        fig.savefig("fig.png")