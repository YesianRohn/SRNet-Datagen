# -*- coding: utf-8 -*-
"""
SRNet data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import math
import numpy as np
import pygame
from pygame import freetype
import random
import queue
import Augmentor

from . import render_text_mask
from . import colorize
from . import data_cfg
import matplotlib.pyplot as plt 

from scipy.ndimage import map_coordinates, gaussian_filter

def assign_ids_to_masks(text_arr, coords):
    """
    Assign unique IDs to each character mask within the bounding boxes.

    Args:
        text_arr: The input mask image (2D array).
        coords: A 2x4xn matrix, where each bounding box is represented by 4 vertices.

    Returns:
        A mask image where each character's mask has a unique ID.
    """
    # Step 1: Threshold the text_arr to create a binary image
    binary_mask = np.where(text_arr > 0, 255, 0).astype(np.uint8)

    # Create an output mask with the same shape as text_arr
    output_mask = np.zeros_like(binary_mask, dtype=np.int32)

    n = coords.shape[2]  # Number of bounding boxes

    for i in range(n):
        # Get the 4 vertices of the bounding box
        box = coords[:, :, i]
        pts = box.T.astype(int)  # Convert to integer type
        
        # Create a mask for the current bounding box
        bbox_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.fillPoly(bbox_mask, [pts], 255)  # Fill the bounding box area with 255

        # Extract the intersection of the bounding box mask and the binary mask
        char_mask = cv2.bitwise_and(binary_mask, bbox_mask)

        # Assign the ID to the character mask (i+1)
        output_mask[char_mask == 255] = i + 1

    return output_mask

class datagen():

    def __init__(self):
        
        freetype.init()
        
        self.hard_font_list = [os.path.join(data_cfg.hard_font, font_name) for font_name in os.listdir(data_cfg.hard_font)]
        self.normal_font_list = [os.path.join(data_cfg.normal_font, font_name) for font_name in os.listdir(data_cfg.normal_font)]
        self.font_cache = {}
        
        color_filepath = data_cfg.color_filepath
        self.colorsRGB, self.colorsLAB = colorize.get_color_matrix(color_filepath)
        
        text_filepath = data_cfg.text_filepath
        self.text_list = open(text_filepath, 'r').readlines()
        self.text_list = [text.strip() for text in self.text_list]

        self.chars = list("0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~!\"#$%&'()*+,-./")
        
        bg_filepath = data_cfg.bg_filepath
        self.bg_list = []
        for img_file in os.listdir(bg_filepath):
            full_path = os.path.join(bg_filepath, img_file)
            if os.path.isfile(full_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.bg_list.append(full_path)
        
        self.surf_augmentor = Augmentor.DataPipeline(None)
        self.surf_augmentor.random_distortion(probability = data_cfg.elastic_rate,
            grid_width = data_cfg.elastic_grid_size, grid_height = data_cfg.elastic_grid_size,
            magnitude = data_cfg.elastic_magnitude)
        
        self.bg_augmentor = Augmentor.DataPipeline(None)
        self.bg_augmentor.random_brightness(probability = data_cfg.brightness_rate, 
            min_factor = data_cfg.brightness_min, max_factor = data_cfg.brightness_max)
        self.bg_augmentor.random_color(probability = data_cfg.color_rate, 
            min_factor = data_cfg.color_min, max_factor = data_cfg.color_max)
        self.bg_augmentor.random_contrast(probability = data_cfg.contrast_rate, 
            min_factor = data_cfg.contrast_min, max_factor = data_cfg.contrast_max)

    def get_cached_font(self, font_path):
        if font_path not in self.font_cache:
            self.font_cache[font_path] = freetype.Font(font_path)
        return self.font_cache[font_path]


    def gen_srnet_data_with_background(self):
        
        while True:
            # choose font, text and bg
            if np.random.rand() < data_cfg.hard_rate:
                font = np.random.choice(self.hard_font_list)
            else:
                font = np.random.choice(self.normal_font_list)

            ### Edit
            # text1, text2 = np.random.choice(self.text_list), np.random.choice(self.text_list)

            ### Synthesis
            text1 = np.random.choice(self.text_list)
            text2 = list(text1)
            num_to_replace = max(1, len(text1) // 2)
            replace_indices = np.random.choice(len(text1), num_to_replace, replace=False)
            for idx in replace_indices:
                text2[idx] = np.random.choice(self.chars)
            text2 = ''.join(text2)
            
            upper_rand = np.random.rand()
            if upper_rand < data_cfg.capitalize_rate + data_cfg.uppercase_rate:
                text1, text2 = text1.capitalize(), text2.capitalize()
            if upper_rand < data_cfg.uppercase_rate:
                text1, text2 = text1.upper(), text2.upper()
            bg = cv2.imread(random.choice(self.bg_list))

            # init font
            font = self.get_cached_font(font)
            # font = freetype.Font(font)
            font.antialiased = True
            font.origin = True

            # choose font style
            font.size = np.random.randint(data_cfg.font_size[0], data_cfg.font_size[1] + 1)
            font.underline = np.random.rand() < data_cfg.underline_rate
            font.strong = np.random.rand() < data_cfg.strong_rate
            font.oblique = np.random.rand() < data_cfg.oblique_rate

            # render text to surf
            param = {
                        'is_curve': np.random.rand() < data_cfg.is_curve_rate,
                        'curve_rate': data_cfg.curve_rate_param[0] * np.random.randn() 
                                      + data_cfg.curve_rate_param[1],
                        'curve_center': np.random.randint(0, len(text1))
                    }
            surf1, bbs1 = render_text_mask.render_text(font, text1, param)
            param['curve_center'] = int(param['curve_center'] / len(text1) * len(text2))
            surf2, bbs2 = render_text_mask.render_text(font, text2, param)

            del font

            # get padding
            padding_ud = np.random.randint(data_cfg.padding_ud[0], data_cfg.padding_ud[1] + 1, 2)
            padding_lr = np.random.randint(data_cfg.padding_lr[0], data_cfg.padding_lr[1] + 1, 2)
            padding = np.hstack((padding_ud, padding_lr))

            # perspect the surf
            rotate = data_cfg.rotate_param[0] * np.random.randn() + data_cfg.rotate_param[1]
            zoom = data_cfg.zoom_param[0] * np.random.randn(2) + data_cfg.zoom_param[1]
            shear = data_cfg.shear_param[0] * np.random.randn(2) + data_cfg.shear_param[1]
            perspect = data_cfg.perspect_param[0] * np.random.randn(2) +data_cfg.perspect_param[1]
            surf1, bbs1 = render_text_mask.perspective(surf1, bbs1, rotate, zoom, shear, perspect, padding) # w first
            surf2, bbs2 = render_text_mask.perspective(surf2, bbs2, rotate, zoom, shear, perspect, padding) # w first

            # choose a background
            surf1_h, surf1_w = surf1.shape[:2]
            surf2_h, surf2_w = surf2.shape[:2]
            surf_h = max(surf1_h, surf2_h)
            surf_w = max(surf1_w, surf2_w)
            surf1, bbs1 = render_text_mask.center2size(surf1, (surf_h, surf_w), bbs1)
            surf2, bbs2 = render_text_mask.center2size(surf2, (surf_h, surf_w), bbs2)

            bg_h, bg_w = bg.shape[:2]
            if bg_w < surf_w or bg_h < surf_h:
                continue
            x = np.random.randint(0, bg_w - surf_w + 1)
            y = np.random.randint(0, bg_h - surf_h + 1)
            t_b = bg[y:y+surf_h, x:x+surf_w, :]
            
            # augment surf
            surfs = [[surf1, surf2]]
            self.surf_augmentor.augmentor_images = surfs
            surf1, surf2 = self.surf_augmentor.sample(1)[0]
            
            # bg augment
            bgs = [[t_b]]
            self.bg_augmentor.augmentor_images = bgs
            t_b = self.bg_augmentor.sample(1)[0][0]

            # get min h of bbs
            min_h1 = np.min(bbs1[:, 3])
            min_h2 = np.min(bbs2[:, 3])
            min_h = min(min_h1, min_h2)

            # get font color
            if np.random.rand() < data_cfg.use_random_color_rate:
                fg_col, bg_col = (np.random.rand(3) * 255.).astype(np.uint8), (np.random.rand(3) * 255.).astype(np.uint8)
            else:
                fg_col, bg_col = colorize.get_font_color(self.colorsRGB, self.colorsLAB, t_b)

            # colorful the surf and conbine foreground and background
            param = {
                        'is_border': np.random.rand() < data_cfg.is_border_rate,
                        'bordar_color': tuple(np.random.randint(0, 256, 3)),
                        'is_shadow': np.random.rand() < data_cfg.is_shadow_rate,
                        'shadow_angle': np.pi / 4 * np.random.choice(data_cfg.shadow_angle_degree)
                                        + data_cfg.shadow_angle_param[0] * np.random.randn(),
                        'shadow_shift': data_cfg.shadow_shift_param[0, :] * np.random.randn(3)
                                        + data_cfg.shadow_shift_param[1, :],
                        'shadow_opacity': data_cfg.shadow_opacity_param[0] * np.random.randn()
                                          + data_cfg.shadow_opacity_param[1]
                    }
            _, i_s = colorize.colorize(surf1, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            _, i_t = colorize.colorize(surf2, t_b, fg_col, bg_col, self.colorsRGB, self.colorsLAB, min_h, param)
            # skeletonization
            surf1 = assign_ids_to_masks(surf1, bbs1)
            surf2 = assign_ids_to_masks(surf2, bbs2)
            break
   
        return [i_s, surf1, i_t, surf2, t_b, text1, text2]

