from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def solarize(image, threshold=128):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(img)))

    img = image.copy()
    idx = img >= threshold
    img[idx] = 255 - image[idx]
    return img


class Solarize(BaseTransform):
    def __init__(self, threshold=128):
        self.threshold = threshold

    def image_transform(self, image):
        return solarize(image, self.threshold)


class RandomSolarize(BaseRandomTransform):
    def __init__(self, threshold=128, prob=0.5):
        self.threshold = threshold
        self.prob      = prob

    def image_transform(self, image):
        return solarize(image, self.threshold)