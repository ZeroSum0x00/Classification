import cv2
import numpy as np


def random_range(a=0, b=1):
    return np.random.rand() * (b - a) + a


def change_color_space(image, current_space="BGR", to_space="BGR"):
    if not ((current_space.lower() in {'bgr', 'rgb', 'hsv', 'gray'}) 
             and (to_space.lower() in {'bgr', 'rgb', 'hsv', 'gray'})):
        raise NotImplementedError
    if current_space.lower() != 'gray' or (image.shape[-1] != 1 and len(image.shape) > 2):
        if current_space.lower() == 'bgr' and to_space.lower() == 'rgb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if current_space.lower() == 'rgb' and to_space.lower() == 'bgr':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif current_space.lower() == 'bgr' and to_space.lower() == 'hsv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image
