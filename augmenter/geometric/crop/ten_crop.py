import numbers
from augmenter.base_transform import BaseTransform
from .five_crop import five_crop
from ..flip import vflip, hflip
from utils.auxiliary_processing import is_numpy_image



def ten_crop(image, size, vertical_flip=False):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(image, size)

    if vertical_flip:
        image = vflip(image)
    else:
        image = hflip(image)

    second_five = five_crop(image, size)
    return first_five + second_five


class TenCrop(BaseTransform):

    """Crop the given image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def image_transform(self, image):
        return ten_crop(image, self.size, self.vertical_flip)
    