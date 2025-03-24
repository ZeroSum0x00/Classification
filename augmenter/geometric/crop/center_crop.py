import numbers
from .crop import crop
from ..pad import pad
from augmenter.base_transform import BaseTransform


def center_crop(image, size, fill_color=0, padding_mode='constant'):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    crop_height, crop_width = size
    image_height, image_width = image.shape[:2]

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        image = pad(image, padding_ltrb, fill_color=fill_color, padding_mode=padding_mode)
        image_height, image_width = image.shape[:2]

        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(image, crop_top, crop_left, crop_height, crop_width)


class CenterCrop(BaseTransform):

    """Crops the given image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, fill_color=0, padding_mode='constant'):
        self.size         = size
        self.fill_color   = fill_color
        self.padding_mode = padding_mode

    def image_transform(self, image):
        return center_crop(image, self.size, fill_color=self.fill_color, padding_mode=self.padding_mode)