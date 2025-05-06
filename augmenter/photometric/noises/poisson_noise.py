import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform


def poisson_noise(image):
    imgtype = image.dtype
    image = image.astype(np.float32)/255.0
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(image.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy.astype(imgtype)


class PoissonNoise(BaseTransform):

    def image_transform(self, image):
        return poisson_noise(image)


class RandomPoissonNoise(BaseRandomTransform):

    def image_transform(self, image):
        return poisson_noise(image)
    