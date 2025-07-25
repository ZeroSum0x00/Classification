import io
import cv2
import requests
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image
from utils.logger import logger



def blend_random_image(image, ratio=0.8):
    URL = "https://picsum.photos/{}/{}/?random"

    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
    
    h, w = image.shape[:2]
    try:
        r = requests.get(URL.format(w, h), allow_redirects=True)
        f = io.BytesIO(r.content)
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        random_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.addWeighted(image, ratio, random_img, 1 - ratio, 0)
    except requests.exceptions.ConnectionError as e:
        logger.error("Unable to download image. Error: {}".format(e))
    except Exception as e:
        logger.error("Unknown error occurred '{}'".format(e))


class BlendRandomImage(BaseTransform):
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def image_transform(self, image):
        return blend_random_image(image, self.ratio)


class RandomBlendRandomImage(BaseRandomTransform):
    def __init__(self, ratio=0.8, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def image_transform(self, image):
        return blend_random_image(image, self.ratio)
    