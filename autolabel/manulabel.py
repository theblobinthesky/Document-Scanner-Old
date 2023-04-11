#!/usr/bin/python3
from glob import glob
import pygame
import cv2
import numpy as np
from support import get_best_segmentation, get_single_accurate_segmentation, unsharpen_mask
import utils
from pathlib import Path

min_close_distance = 20
fix_radius = 5
mask_opacity = 0.4
processing_size = 0.5

def scale_mask_up(mask, size):
    mask = cv2.resize(mask, size)
    return (mask == 255).astype("uint8") * 255


def render_image_to_be_annotated(window, img, mask):                
    window.fill((255, 255, 255))

    if mask_done:
        img_f = img.astype(np.float32) / 255.0
        mask_f = mask.astype(np.float32) / 255.0
        mask_f = np.repeat(mask_f[:, :, np.newaxis], 3, 2)
        
        comp = (1.0 - mask_opacity) * img_f + mask_opacity * mask_f
        surf = pygame.surfarray.make_surface(comp * 255)
        window.blit(surf, (0, 0))
    else:
        surf = pygame.surfarray.make_surface(img)
        window.blit(surf, (0, 0))


scans = utils.get_scans()
for scan_dir in scans:
    print()
    print(f"Working on '{scan_dir}'...")

    paths = glob(f"{scan_dir}/*.jpg")

    if len(paths) == 0:
        print("Skipping. Please run autolabel on this scan first.")
        continue

    best_path, _, _ = get_best_segmentation(paths)
    name = Path(best_path).name


    img = cv2.imread(best_path)
    size = img.shape[0:2][::-1]
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = unsharpen_mask(img)
    
    width, height, _ = img.shape
    mask_done, mask = False, None

    points, fix_points = [], []
    closed, fixed = False, False

    window = pygame.display.set_mode((width, height))
    while True:
        mouse_down = False
        quit_this_one = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Skipping by user demand.")
                quit_this_one = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == pygame.MOUSEWHEEL and mask_done:
                mask_dir = f"{scan_dir}/{utils.annotation_subdir}"
                mask_path = f"{mask_dir}/{name}"
                utils.make_dirs([mask_dir])
                cv2.imwrite(mask_path, scale_mask_up(mask, size))
                quit_this_one = True
            
        if quit_this_one: break

        render_image_to_be_annotated(window, img, mask)

        # render and handle the annotation
        pressed = pygame.mouse.get_pressed()
        mouse_pos = np.array(pygame.mouse.get_pos())
        
        if pressed[2] and mouse_down:
            if len(fix_points) > 0: 
                fix_points = fix_points[:-1]
                fixed = False
            elif len(points) > 0: 
                points = points[:-1]
                closed = False
                mask_done = False

        points_not_empty = len(points) > 0
        
        if pressed[0] and mouse_down:
            if closed:
                fix_points.append(mouse_pos)
            else:
                if points_not_empty:
                    dist = np.linalg.norm(points[0] - mouse_pos)
                else:
                    dist = min_close_distance + 1.0
                    
                if dist < min_close_distance:
                    closed = True
                else:
                    points.append(mouse_pos)
        
        if pressed[1] and mouse_down and closed:
            mask = get_single_accurate_segmentation(img, np.array(points), fix_points, fix_radius)
            if mask_done: fixed = True
            mask_done = True
            pass


        # draw visuals
        if points_not_empty and not closed:
            pos = points[-1]
            pygame.draw.line(window, (255, 0, 0), (pos[0], pos[1]), (mouse_pos[0], mouse_pos[1]), 3)


        for i in range(len(points) - 1):
            pygame.draw.line(window, (0, 0, 0), (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), 3)
            
        if closed:
            pygame.draw.line(window, (0, 0, 0), (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), 3)


        if points_not_empty:
            pygame.draw.circle(window, (200, 200, 200), points[0], min_close_distance)

        for i in range(len(points)):
            pygame.draw.circle(window, (0, 0, 0), points[i], 5)       

        
        for pt in fix_points:
            pygame.draw.circle(window, (255, 0, 0), pt, fix_radius)             
                
        pygame.display.flip()