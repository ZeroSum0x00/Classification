import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Rotate:
    def __init__(self, angle, padding_color=None):
        self.angle = angle
        self.padding_color = padding_color
        
    def __call__(self, image):
        h, w, _ = image.shape
        a,b = w/2, h/2
        M = cv2.getRotationMatrix2D((a, b), self.angle, 1)
        if self.padding_color:
            if isinstance(self.padding_color, int):
                fill_color = [self.padding_color, self.padding_color, self.padding_color]
            else:
                fill_color = self.padding_color
        else:
            fill_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=fill_color)
        return image


class RandomRotate:
    def __init__(self, angle_range=15, prob=0.5, padding_color=None):
        self.prob       = prob
        angle = np.random.randint(-angle_range, angle_range)
        self.aug        = Rotate(angle, padding_color=padding_color)
        
    def __call__(self, image):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image = self.aug(image)
        return image