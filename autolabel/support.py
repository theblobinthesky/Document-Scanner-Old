import sys
sys.path.insert(0, '../model')
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import model as m
import seg_model as sm
from tqdm import tqdm
from utils import make_dirs, get_scan_label_dir, get_unrotated_img_paths
import shutil

model_path = "../model/models/main_seg_2.pth"
model_size, processing_size = 64, 256
processing_size = (processing_size, processing_size)
binarize_threshold = 0.5
mask_size_kernels = (3, 3)
mask_size_factors = (0.95, 1.01)
accurate_mask_size_factors = (0.98, 1.05)
min_ratio = 0.95

model = sm.SegModel()
model.load_state_dict(torch.load(model_path))
model = model.to(device=m.device)

def get_segmask_from_model(frame):
    with torch.no_grad():
        frame = frame.astype("float32") / 255.0
        ten = torch.from_numpy(frame).to(device=m.device)
        ten = ten.permute(2, 0, 1).unsqueeze(0)

        segmask = model(ten)
        segmask = segmask.squeeze(0).permute((1, 2, 0))
    return segmask.detach().cpu().numpy()

def get_largest_contour(mask):
    contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda contour: cv2.contourArea(contour))
    contour = contour.reshape((-1, 2))
    return contour
            
def get_mask_from_contour(contour, size):
    contour = contour.astype("int32")
    mask = np.zeros(size, np.uint8)
    mask = cv2.fillPoly(mask, [contour], (255))
    return mask
            
def get_simple_contour(contour, fac=0.12):
    contour = cv2.approxPolyDP(contour, fac * cv2.arcLength(contour, True), True)
    contour = contour.reshape((-1, 2))

    return contour

def isolate_largest_mask(mask):
    contour = get_largest_contour(mask)
    mask = get_mask_from_contour(contour, mask.shape[0:2])
    return mask


def unsharpen_mask(frame, sigma=2.0, alpha=2.0):
    gaussian_3 = cv2.GaussianBlur(frame, (0, 0), sigma)
    unsharp_image = cv2.addWeighted(frame, alpha, gaussian_3, -1.0, 0)
    return unsharp_image

def grab_cut(img, pr_fg_mask, fg_mask, bg_mask, size, iters=6):
    fg_model, bg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            
    mask = np.zeros(size, np.uint8)
    mask[bg_mask == 0] = cv2.GC_PR_BGD
    mask[bg_mask == 255] = cv2.GC_BGD
    mask[pr_fg_mask == 255] = cv2.GC_PR_FGD
    mask[fg_mask == 255] = cv2.GC_FGD

    cv2.grabCut(img, mask, None, bg_model, fg_model, iters, cv2.GC_INIT_WITH_MASK)       
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    return get_largest_contour(mask)

def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_best_segmentation(paths):
    contours = []
    
    for path in tqdm(paths, desc="Dense segmentation"):
        img = read_img(path)
        original_size = img.shape[0:2]
        img = cv2.resize(img, processing_size)
        small_frame = cv2.resize(img, (model_size, model_size))

        pr_fg_mask = get_segmask_from_model(small_frame)
        pr_fg_mask = (pr_fg_mask > binarize_threshold).astype("uint8") * 255
        pr_fg_mask = isolate_largest_mask(pr_fg_mask)

                
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_size_kernels[0], mask_size_kernels[0]))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_size_kernels[1], mask_size_kernels[1]))
        fg_mask, bg_mask = cv2.erode(pr_fg_mask, small_kernel), cv2.dilate(pr_fg_mask, large_kernel)        
        bg_mask = cv2.bitwise_not(bg_mask)

        # resize them to be of frame size, not small frame size
        def upscale_small_mask(small):
            large = cv2.resize(small, processing_size)
            return (large == 255).astype("uint8") * 255

        pr_fg_mask, fg_mask, bg_mask = upscale_small_mask(pr_fg_mask), upscale_small_mask(fg_mask), upscale_small_mask(bg_mask)

        # unshapen mask frame
        img = unsharpen_mask(img, sigma=4.0, alpha=2.0)


        # grabcut the fine contours
        contour = grab_cut(img, pr_fg_mask, fg_mask, bg_mask, processing_size)
        simple_contour = get_simple_contour(contour)
        mask, simple_mask = get_mask_from_contour(contour, processing_size), get_mask_from_contour(simple_contour, processing_size)

        conj = cv2.bitwise_and(mask, cv2.bitwise_not(simple_mask))
        ratio = conj.sum() / simple_mask.sum()

        if len(simple_contour) != 4:
            ratio = 0.0

        # surf = pygame.surfarray.make_surface(np.rot90(cv2.bitwise_and(img, simple_mask[:, :, np.newaxis].repeat(3, 2))))
        # surf = pygame.transform.scale(surf, screen_size)
        # screen.blit(surf, (0, 0))
        # pygame.display.update()
 
        contour = simple_contour
        contour = contour.astype("float32")
        contour[:, 0] *= original_size[1] / float(processing_size[0])
        contour[:, 1] *= original_size[0] / float(processing_size[1])
        contours.append((ratio, contour))

    value = max(contours, key=lambda value: value[0])
    is_confident = value[0] > min_ratio
    path = paths[contours.index(value)]

    return path, value[1], is_confident

def get_guided_segmentations(paths, contours):
    masks = []

    for i, path in enumerate(tqdm(paths, desc="Guided dense segmentation")):
        img = read_img(path)
        size = img.shape[0:2]

        contour = contours[i]
        middle = contour.mean(0)
        middle_contour = contour - middle
        fg_contour, bg_contour = middle + middle_contour * mask_size_factors[0], middle + middle_contour * mask_size_factors[1]
        pr_fg_mask = get_mask_from_contour(contour, size)
        fg_mask = get_mask_from_contour(fg_contour, size)
        bg_mask = cv2.bitwise_not(get_mask_from_contour(bg_contour, size))

        # unshapen mask frame
        img = unsharpen_mask(img, sigma=8.0, alpha=2.0)

        contour = grab_cut(img, pr_fg_mask, fg_mask, bg_mask, size)
        mask = get_mask_from_contour(contour, size)
        masks.append(mask)

    return masks

def get_single_accurate_segmentation(img, contour, fix_points, fix_radius):
    contour = np.flip(contour, axis=1)
    middle = contour.mean(0)[np.newaxis, :]
    middle_contour = contour - middle
    fg_contour, bg_contour = middle + accurate_mask_size_factors[0] * middle_contour, middle + accurate_mask_size_factors[1] * middle_contour

    fg_mask = get_mask_from_contour(contour, img.shape[:2])
    bg_mask = get_mask_from_contour(bg_contour, img.shape[:2])
    bg_mask = cv2.bitwise_not(bg_mask)

    for pt in fix_points:
        cv2.circle(bg_mask, (pt[1], pt[0]), 100, (255), -1)

    # unshapen mask frame
    img = unsharpen_mask(img, sigma=4.0, alpha=2.0)

    # grabcut the fine contours
    contour = grab_cut(img, fg_mask, fg_mask, bg_mask, img.shape[:2], iters=12)
    mask = get_mask_from_contour(contour, img.shape[:2])
    return mask

def save_images_and_masks(paths, masks, data, image_dir, mask_dir):
    make_dirs([image_dir, mask_dir])

    for i, mask in enumerate(masks):
        path = paths[i]
        name = Path(path).name

        shutil.copyfile(path, f"{image_dir}/{name}")

        mask_path = f"{mask_dir}/{name}"
        cv2.imwrite(mask_path, mask)