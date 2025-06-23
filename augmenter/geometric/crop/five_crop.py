import numbers
from augmenter.base_transform import BaseTransform
from .crop import crop
from .center_crop import center_crop
from utils.auxiliary_processing import is_numpy_image



def five_crop(image, size):
    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    h, w, _ = image.shape
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = crop(image, 0, 0, crop_h, crop_w)
    tr = crop(image, 0, w - crop_w, crop_h, crop_w)
    bl = crop(image, h - crop_h, 0, crop_h, crop_w)
    br = crop(image, h - crop_h, w - crop_w, crop_h, crop_w)
    center = center_crop(image, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


class FiveCrop(BaseTransform):

    """Crop the given image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def image_transform(self, image):
        return five_crop(image, self.size)
