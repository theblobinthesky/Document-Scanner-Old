#!/usr/bin/python3
from model import binarize_threshold, device
from data import load_seg_dataset, load_contour_dataset, load_bm_dataset
from seg_model import points_per_side_incl_start_corner
from torchvision.transforms import Resize
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

num_examples = 8
dpi = 50
circle_radius = 2
rect_color = (0.0, 1.0, 0.0)
circle_color = (1.0, 0.0, 0.0)

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

def apply_feature_to_image(img, center_rel_to_cell, size_rel_to_img, contour, objectness):
    img_size = objectness.shape[:2]
    w, h = img_size
    img_size = np.array(img_size, np.float32)

    img = np.ascontiguousarray(img.copy())

    for x in range(w):
        for y in range(h):
            if objectness[x, y, 0] < binarize_threshold:
                continue

            def swap(pt):
                return (pt[1], pt[0])

            center = (np.array([x, y], np.float32) + center_rel_to_cell[x, y]) / img_size
            size = size_rel_to_img[x, y]

            def get_pt(start):
                pt = ((center + contour[x, y, start:start+2]) * 64.0).astype("int32")
                confidence = contour[x, y, start + 2]
                return pt, confidence
        
            contour_pts = [ get_pt(i * 3) for i in range(4 * points_per_side_incl_start_corner) ]

            center = (64.0 * center).astype("int32")
            h_size = (64.0 * size / 2.0).astype("int32")

            cv2.rectangle(img, swap(center - h_size), swap(center + h_size), rect_color)

            for pt, confidence in contour_pts:
                if confidence > binarize_threshold:
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

        x, (feature_label, _) = model.input_and_label_from_dict(dict)
        feature_pred = model(x)

        unpack_label = model.unpack_feature(feature_label)
        unpack_pred = model.unpack_feature(feature_pred)

        x = cpu(x)
        unpack_label = [cpu(u) for u in unpack_label]
        unpack_pred = [cpu(u) for u in unpack_pred]

        xs.append(x)
        ys.append(unpack_label)
        preds.append(unpack_pred)


    title = ["contour pred", "contour label"]

    fig = plt.figure(figsize=(25, 25), dpi=dpi)
    i = 0

    for e in range(num_examples):
        img = xs[e][:, :, :3]
        
        unpack_pred = preds[e]
        unpack_label = ys[e]

        center_rel_to_cell_label, size_label, contour_label, objectness_label = unpack_label[0], unpack_label[1], unpack_label[2], unpack_label[3]
        center_rel_to_cell_pred, size_pred, contour_pred, objectness_pred = unpack_pred[0], unpack_pred[1], unpack_pred[2], unpack_pred[3]
        
        list = [
            apply_feature_to_image(img, center_rel_to_cell_pred, size_pred, contour_pred, objectness_pred), 
            apply_feature_to_image(img, center_rel_to_cell_label, size_label, contour_label, objectness_label)
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
    cap = cv2.VideoCapture(0)
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
   
        feature = model(img)
        unpack = model.unpack_feature(feature)
        unpack = [cpu(u) for u in unpack]
        [center_rel_to_cell, size, contour, objectness] = unpack

        frame = apply_feature_to_image(frame, center_rel_to_cell, size, contour, objectness)

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
    model.load_state_dict(torch.load("models/contour_model.pth"))
    model = model.to(device=device)

    benchmark_cam_contour_model(model)
    # fig = benchmark_plt_contour(model)
    # fig.savefig("fig.png")