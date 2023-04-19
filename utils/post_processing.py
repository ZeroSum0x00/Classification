import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from utils.logger import logger


def get_labels(label_file):
    with open(label_file, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def resize_image(image, target_size, letterbox_image):
    h, w, _    = image.shape
    ih, iw, _  = target_size
    if letterbox_image:
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=image.dtype)
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        return image_paded
    else:
        image = cv2.resize(image, (iw, ih))
        return image


def preprocess_input(image):
    image /= 255.0
    return image