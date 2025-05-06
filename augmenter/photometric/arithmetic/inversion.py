from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def inversion(image):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(img)))
    img = image.copy()
    img[:, :, :3] = 255 - img[:, :, :3]
    return img


class Inversion(BaseTransform):

    def image_transform(self, image):
        return inversion(image)

class RandomInversion(BaseRandomTransform):

    def image_transform(self, image):
        return inversion(image)
