import cv2

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def jpeg_noise(image, quality=0.1):
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(100 * quality)])
    return cv2.imdecode(buff, cv2.IMREAD_COLOR)


class JpegNoise(BaseTransform):
    def __init__(self, quality=0.1):
        self.quality = quality

    def image_transform(self, image):
        return jpeg_noise(image, self.quality)


class RandomJpegNoise(BaseRandomTransform):
    def __init__(self, quality=0.1, prob=0.5):
        self.quality = quality
        self.prob = prob

    def image_transform(self, image):
        return jpeg_noise(image, self.quality)
