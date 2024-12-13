from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def posterize(image, bits):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(img)))

    img = image.copy()

    if bits >= 8:
        return img
        
    mask = ~(2 ** (8 - bits) - 1)
    img = img & mask
    return img


class Posterize(BaseTransform):
    def __init__(self, bits):
        self.bits = bits

    def image_transform(self, image):
        return posterize(image, self.bits)


class RandomPosterize(BaseRandomTransform):
    def __init__(self, bits, prob=0.5):
        self.bits = bits
        self.prob = prob

    def image_transform(self, image):
        return posterize(image, self.bits)