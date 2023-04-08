#!/usr/bin/python3
from glob import glob
import cv2
import numpy as np
import torch
from seg_model import SegModel
from model import device
import pygame
from tqdm import tqdm
import time

dir = "/media/shared/Projekte/DocumentScanner/datasets/Custom"
model_path = "models/main_seg_2.pth"
processing_size = 256
model_size = 64
mask_size_kernels = (15, 7)
mask_size_factors = (0.9, 1.1)
binarize_threshold = 0.8
max_ratio = 0.03
track_box_size = 51
hbs = track_box_size // 2
iters = 4


paths = ["/media/shared/Projekte/DocumentScanner/datasets/Custom/20230319_175025.mp4"] # 20230407_184309
paths = glob(f"{dir}/*.mp4")

model = SegModel()
model.load_state_dict(torch.load(model_path))
model = model.to(device=device)

pygame.init()

def visualize_track(start_index, track):
    for i in range(len(track)):
        if track[i] == None: break

        fi = start_index + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        frame = read_frame(cap)

        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        for box in track[i]:
            if box == None: continue
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cx, cy = int(round(x + w / 2.0)), int(round(round(y + h / 2.0)))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.circle(frame, (cx, cy), 5, color, -1)


        surf = pygame.surfarray.make_surface(np.rot90(frame))
        surf = pygame.transform.scale(surf, screen_size)

        screen.blit(surf, (0, 0))
        pygame.display.update()

        # check for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                quit()


def get_segmask_from_model(frame):
    with torch.no_grad():
        frame = frame.astype("float32") / 255.0
        ten = torch.from_numpy(frame).to(device=device)
        ten = ten.permute(2, 0, 1).unsqueeze(0)

        segmask = model(ten)
        segmask = segmask.squeeze(0).permute((1, 2, 0))
    return segmask.detach().cpu().numpy()


def read_frame(cap):
    _, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (processing_size, processing_size))

    return frame

def get_largest_contour(mask):
    contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda contour: cv2.contourArea(contour))
    return contour
            
def get_mask_from_contour(contour, size):
    contour = contour.astype("int32")
    mask = np.zeros(size, np.uint8)
    mask = cv2.fillPoly(mask, [contour], (255))
    return mask
            
def get_simple_contour(contour, fac=0.12):
    return cv2.approxPolyDP(contour, fac * cv2.arcLength(contour, True), True)

def isolate_largest_mask(mask):
    contour = get_largest_contour(mask)
    mask = get_mask_from_contour(contour, mask.shape[0:2])
    return mask


def unsharpen_mask(frame, sigma=2.0, alpha=2.0):
    gaussian_3 = cv2.GaussianBlur(frame, (0, 0), sigma)
    unsharp_image = cv2.addWeighted(frame, alpha, gaussian_3, -1.0, 0)
    return unsharp_image

def grab_cut(frame, pr_fg_mask, fg_mask, bg_mask, size):
    fg_model, bg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            
    mask = np.zeros(size, np.uint8)
    mask[bg_mask == 0] = cv2.GC_PR_BGD
    mask[bg_mask == 255] = cv2.GC_BGD
    mask[pr_fg_mask == 255] = cv2.GC_PR_FGD
    mask[fg_mask == 255] = cv2.GC_FGD
    cv2.grabCut(frame, mask, None, bg_model, fg_model, 6, cv2.GC_INIT_WITH_MASK)       
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    return get_largest_contour(mask)

def do_dense_segmentation(cap, frames):
    contours_enum = []
    
    for i in tqdm(range(frames), desc="Dense segmentation"):
        frame = read_frame(cap)
        small_frame = cv2.resize(frame, (model_size, model_size))


        pr_fg_mask = get_segmask_from_model(small_frame)
        pr_fg_mask = (pr_fg_mask > binarize_threshold).astype("uint8") * 255
        pr_fg_mask = isolate_largest_mask(pr_fg_mask)

                
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_size_kernels[0], mask_size_kernels[0]))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_size_kernels[1], mask_size_kernels[1]))
        fg_mask, bg_mask = cv2.erode(pr_fg_mask, small_kernel), cv2.dilate(pr_fg_mask, large_kernel)        
        bg_mask = cv2.bitwise_not(bg_mask)

        # resize them to be of frame size, not small frame size
        def upscale_small_mask(small):
            large = cv2.resize(small, frame.shape[0:2])
            return (large == 255).astype("uint8") * 255

        pr_fg_mask, fg_mask, bg_mask = upscale_small_mask(pr_fg_mask), upscale_small_mask(fg_mask), upscale_small_mask(bg_mask)

        # unshapen mask frame
        frame = unsharpen_mask(frame, sigma=4.0, alpha=2.0)


        # grabcut the fine contours
        contour = grab_cut(frame, pr_fg_mask, fg_mask, bg_mask, frame.shape[0:2])
        simple_contour = get_simple_contour(contour)
        mask, simple_mask = get_mask_from_contour(contour, frame.shape[0:2]), get_mask_from_contour(simple_contour, frame.shape[0:2])

        conj = cv2.bitwise_and(mask, cv2.bitwise_not(simple_mask))
        ratio = conj.sum() / simple_mask.sum()
        if len(simple_contour) != 4 or ratio > max_ratio:
            continue

        surf = pygame.surfarray.make_surface(np.rot90(cv2.bitwise_and(frame, simple_mask[:, :, np.newaxis].repeat(3, 2))))
        surf = pygame.transform.scale(surf, screen_size)
        screen.blit(surf, (0, 0))
        pygame.display.update()
            
        simple_contour = simple_contour.reshape((-1, 2))
        contours_enum.append((i, simple_contour))

    return contours_enum

def do_fused_track(cap, contours_enum):
    tracks = []
    trackers = [ cv2.TrackerCSRT_create() for _ in range(4) ]


    def update_trackers_and_get_boxes(frame):
        boxes = []
        for i in range(4):
            success, box = trackers[i].update(frame)
        
            if success: boxes.append(box)
            else: boxes.append(None)
            
        return boxes


    for i, (frame_index, simple_contour) in enumerate(tqdm(contours_enum, desc="Tracking")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame = read_frame(cap)
        
        boxes = [(x - hbs, y - hbs, track_box_size, track_box_size) for (x, y) in simple_contour]
        for t in range(4):
            trackers[t].init(frame, boxes[t])
            

        # track forward
        if i == len(contours_enum) - 1:
            next_frame_index = frames - 1
        else:
            next_frame_index = contours_enum[i + 1][0]

        fw_track = [boxes]
        for fw_frame_index in range(frame_index + 1, next_frame_index + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fw_frame_index)
            fw_frame = read_frame(cap)
            fw_track.append(update_trackers_and_get_boxes(frame))


        # track backward
        if i == 0:
            last_frame_index = 0
        else:
            last_frame_index = contours_enum[i - 1][0]
            
        bw_track = [boxes]
        for bw_frame_index in range(last_frame_index, frame_index).__reversed__():
            cap.set(cv2.CAP_PROP_POS_FRAMES, bw_frame_index)
            bw_frame = read_frame(cap)
            bw_track.append(update_trackers_and_get_boxes(frame))
            
        bw_track.reverse()


        tracks.append(((last_frame_index, frame_index, next_frame_index), fw_track, bw_track))


    # fuse the forward and backward tracking
    fused_track = []

    def add_to_fused_track(fwt, bwt):
        if fwt == None: fwt = bwt
        if bwt == None: bwt = fwt

        roll_errors = []
        for roll in range(4):
            error = 0.0
            for i in range(len(fwt)):
                f, b = np.array(fwt[i]), np.roll(np.array(bwt[i]), roll, axis=0)
                error += np.abs(f - b).sum()

            roll_errors.append(error)

        roll = roll_errors.index(min(roll_errors))

        for i in range(len(fwt) - 1):
            t = i / float(len(fwt) - 1)
            f, b = np.array(fwt[i]), np.roll(np.array(bwt[i]), roll, axis=0)
            fused = b * t + f * (1.0 - t)
            fused_track.append(fused.tolist())


    if len(tracks) > 0:
        _, _, c_bwt = tracks[0]
        add_to_fused_track(None, c_bwt)
    
        for t in range(len(tracks) - 1):
            _, c_fwt, _ = tracks[t]
            _, _, n_bwt = tracks[t + 1]
                
            add_to_fused_track(c_fwt, n_bwt)
    
        _, c_fwt, _ = tracks[-1]
        add_to_fused_track(c_fwt, None)
    else:
        print("error!")
        exit()

    return fused_track



for path in paths:
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = 30
    start_time = time.time()

    screen_size = (int(width * 0.5), int(height * 0.5))
    screen = pygame.display.set_mode(screen_size)

    print()
    print(f"Working on '{path}'...")

    if False:
        with open(f"{path}.data", "rb") as f:
            data = np.load(f, allow_pickle=True).item()
            fused_track = data["track"]

        if False:
            visualize_track(0, fused_track)

            continue


    for iter in range(1):
        print(f"Working on iteration {iter}...")

        contours_enum = do_dense_segmentation(cap, frames)
        fused_track = do_fused_track(cap, contours_enum)

        # Guided dense segmentation
        contours = []
        for frame_index in tqdm(range(len(fused_track)), desc="Guided dense segmentation"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame = read_frame(cap)

            corners = []
            for box in fused_track[frame_index]: 
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                corners.append((x + w / 2.0, y + h / 2.0))

            corners = np.array(corners)
            
            middle = corners.mean(0)[np.newaxis, :]
            middle_corners = corners - middle
            small_contour = middle + middle_corners * mask_size_factors[0]
            large_contour = middle + middle_corners * mask_size_factors[1]


            pr_fg_mask = get_mask_from_contour(corners, frame.shape[0:2])
            fg_mask = get_mask_from_contour(small_contour, frame.shape[0:2])
            bg_mask = get_mask_from_contour(large_contour, frame.shape[0:2])  
            bg_mask = cv2.bitwise_not(bg_mask)


            # unshapen mask frame
            frame = unsharpen_mask(frame, sigma=4.0, alpha=2.0)


            # grabcut the fine contours
            contour = grab_cut(frame, pr_fg_mask, fg_mask, bg_mask, frame.shape[0:2])
            contour = get_simple_contour(contour, 0.001)

            contours.append(contour)


    # save to disk
    data = {
        "track": fused_track,
        "contours": contours
    }

    with open(f"{path}.data", "wb") as f:
        np.save(f, data, allow_pickle=True)


    runtime = time.time() - start_time
    runtime /= 60.0
    print(f"Finished in {runtime:.2f}min")

    continue


    print("Visualizing forward tracks...")

    for (_, frame_index, _), fw_track, _ in tracks:
        visualize_track(frame_index, fw_track)

    print("Visualizing backward tracks...")

    for (last_frame_index, _, _), _, bw_track in tracks:
        visualize_track(last_frame_index, bw_track)


    cap.release()