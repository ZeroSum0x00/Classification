import numbers
from .crop import crop
from ..pad import pad
from augmenter.base_transform import BaseTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter



def center_crop(metadata, size, fill_color=0, padding_mode="constant"):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    focus_image = get_focus_image_from_metadata(metadata)

    crop_height, crop_width = size
    image_height, image_width = focus_image.shape[:2]

    pad_left = (crop_width - image_width) // 2 if crop_width > image_width else 0
    pad_top = (crop_height - image_height) // 2 if crop_height > image_height else 0
    pad_right = (crop_width - image_width + 1) // 2 if crop_width > image_width else 0
    pad_bottom = (crop_height - image_height + 1) // 2 if crop_height > image_height else 0

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        metadata = pad(
            metadata,
            padding=(pad_left, pad_top, pad_right, pad_bottom),
            fill_color=fill_color,
            padding_mode=padding_mode
        )
        _focus_image = get_focus_image_from_metadata(metadata)
        image_height, image_width = _focus_image.shape[:2]

    if crop_width == image_width and crop_height == image_height:
        return metadata

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))

    metadata = crop(
        metadata,
        top=crop_top,
        left=crop_left,
        height=crop_height,
        width=crop_width
    )

    return metadata



class CenterCrop(BaseTransform):

    """Crops the given image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(
        self,
        size,
        fill_color=0,
        padding_mode="constant",
    ):
        self.size = size
        self.fill_color = fill_color
        self.padding_mode = padding_mode

    def image_transform(self, metadata):
        return center_crop(
            metadata,
            size=self.size,
            fill_color=self.fill_color,
            padding_mode=self.padding_mode,
        )
    