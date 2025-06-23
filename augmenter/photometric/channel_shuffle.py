import random

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def channel_shuffle(image):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    assert image.shape[2] in [3, 4]
    ch_arr = [0, 1, 2]
    random.shuffle(ch_arr)
    image = image[..., ch_arr]
    return image


class ChannelShuffle(BaseTransform):

    def image_transform(self, image):
        return channel_shuffle(image)


class RandomChannelShuffle(BaseRandomTransform):

    def image_transform(self, image):
        return channel_shuffle(image)
