import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Flip:
    def __init__(self, mode='horizontal'):
        self.mode       = mode

    def __call__(self, image):
        h, w, _ = image.shape
        horizontal_list = ['horizontal', 'h']
        vertical_list   = ['vertical', 'v']
        if self.mode.lower() in horizontal_list:
            image = cv2.flip(image, 1)
        elif self.mode.lower() in vertical_list:
            image = cv2.flip(image, 0)
        return image


class RandomFlip:
    def __init__(self, prob=0.5, mode='horizontal'):
        self.prob       = prob
        self.aug        = Flip(mode=mode)
        
    def __call__(self, image):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image = self.aug(image)
        return image