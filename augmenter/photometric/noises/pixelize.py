import cv2

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def pixelize(image, ratio=.2):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    im_height, im_width = image.shape[:2]
    tmp_w, tmp_h = int(im_width * ratio), int(im_height * ratio)

    image = cv2.resize(image, (tmp_w, tmp_h), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
    return image


class Pixelize(BaseTransform):
    def __init__(self, ratio=.2):
        self.ratio = ratio

    def image_transform(self, image):
        return pixelize(image, self.ratio)

class RandomPixelize(BaseRandomTransform):
    def __init__(self, ratio=.2, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def image_transform(self, image):
        return pixelize(image, self.ratio)
