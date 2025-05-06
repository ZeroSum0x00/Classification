import cv2
import random
import numbers

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from ..pad import pad
from utils.auxiliary_processing import is_numpy_image


def crop(image, top, left, height, width):
    assert is_numpy_image(image), "img should be CV Image. Got {}".format(type(image))
    assert height > 0 and width > 0, "height={} and width={} should greater than 0".format(height, width)

    xmin, ymin, xmax, ymax = round(top), round(left), round(top + height), round(left + width)

    try:
        check_point1 = image[xmin, ymin, ...]
        check_point2 = image[xmax - 1, ymax - 1, ...]
    except IndexError:
        image = cv2.copyMakeBorder(
            image,
            top=-min(0, xmin),
            bottom=max(xmax - image.shape[0], 0),
            left=-min(0, ymin),
            right=max(ymax - image.shape[1], 0),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        
        ymax += -min(0, ymin)
        ymin += -min(0, ymin)
        xmax += -min(0, xmin)
        xmin += -min(0, xmin)

    finally:
        return image[xmin:xmax, ymin:ymax, ...].copy()


class Crop(BaseTransform):

    """Crop the given image to desired size.

    Args:
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.
    """

    def __init__(self, top, left, height, width):
        self.top    = top
        self.left   = left
        self.height = height
        self.width  = width

    def image_transform(self, image):
        return crop(image, self.top, self.left, self.height, self.width)


class RandomCrop(BaseRandomTransform):

    """Crop the given image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill_color (number or tuple or dict, optional): Pixel fill_color value used when 
            the padding_mode is constant. Default is 0. If a tuple of length 3,
            it is used to fill_color R, G, B channels respectively.
        padding_mode (str, optional): Type of padding. Should be: constant,
        edge, reflect or symmetric. Default is constant.
    """

    def __init__(
        self,
        size,
        padding=0,
        pad_if_needed=False,
        fill_color=0,
        padding_mode="constant",
        prob=0.5,
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill_color = fill_color
        self.padding_mode = padding_mode
        self.prob = prob

    @staticmethod
    def get_params(image, size):
        h, w, _ = image.shape
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w

        try:
            i = random.randint(0, h - th)
        except ValueError:
            i = random.randint(h - th, 0)
        try:
            j = random.randint(0, w - tw)
        except ValueError:
            j = random.randint(w - tw, 0)
        return i, j, th, tw

    def image_transform(self, image):
        if self.padding > 0:
            image = pad(
                image=image,
                padding=self.padding,
                fill_color=self.fill_color,
                padding_mode=self.padding_mode,
            )

        # pad the width if needed
        if self.pad_if_needed and image.shape[1] < self.size[1]:
            image = pad(
                image=image,
                padding=(int((1 + self.size[1] - image.shape[1]) / 2), 0),
                fill_color=self.fill_color,
                padding_mode=self.padding_mode,
            )

        # pad the height if needed
        if self.pad_if_needed and image.shape[0] < self.size[0]:
            image = pad(
                image=image,
                padding=(0, int((1 + self.size[0] - image.shape[0]) / 2)),
                fill_color=self.fill_color,
                padding_mode=self.padding_mode,
            )

        top, left, height, width = self.get_params(image, self.size)
        return crop(image, top, left, height, width)
    