import cv2

INTER_MODE = {
    'NEAREST': cv2.INTER_NEAREST, 
    'BILINEAR': cv2.INTER_LINEAR, 
    'BICUBIC': cv2.INTER_CUBIC
}


ALLOW_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
epsilon = 1e-7


