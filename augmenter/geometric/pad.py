import cv2
import numbers
import collections.abc as collections

from augmenter.base_transform import BaseTransform
from utils.auxiliary_processing import is_numpy_image



PAD_MOD = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_DEFAULT,
    "symmetric": cv2.BORDER_REFLECT
}


def pad(
    image,
    padding,
    fill_color=(0, 0, 0),
    padding_mode="constant",
):
    if not is_numpy_image(image):
        raise TypeError("Image should be CV Image. Got {}".format(type(image)))

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill_color, (numbers.Number, str, tuple, list)):
        raise TypeError("Got inappropriate fill_color arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ["constant", "edge", "reflect", "symmetric"], \
        "Padding mode should be either constant, edge, reflect or symmetric"

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill_color, numbers.Number):
        fill_color = (fill_color,) * (2 * len(image.shape) - 3)

    if padding_mode == "constant":
        assert (len(fill_color) == 3 and len(image.shape) == 3) or (len(fill_color) == 1 and len(image.shape) == 2), \
            "channel of image is {} but length of fill_color is {}".format(image.shape[-1], len(fill_color))

    image = cv2.copyMakeBorder(
        src=image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=PAD_MOD[padding_mode],
        value=fill_color,
    )
    return image


class Pad(BaseTransform):

    """Pad the given image on all sides with the given "pad" value.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill_color: Pixel fill_color value for constant fill_color. Default is 0. If a tuple of
            length 3, it is used to fill_color R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill_color
            edge: pads with the last value at the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill_color=0, padding_mode="constant"):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill_color, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding      = padding
        self.fill_color         = fill_color
        self.padding_mode = padding_mode

    def image_transform(self, image):
        return pad(image, self.padding, self.fill_color, self.padding_mode)
    