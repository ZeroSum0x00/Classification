import cv2
import math
import random
import numbers

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata
from utils.bbox_processing import coordinates_converter
from .crop import crop
from .resize import resize



def resize_crop(
    metadata,
    top,
    left,
    height,
    width,
    size
):
    metadata = crop(metadata, top, left, height, width)
    metadata = resize(metadata, size, keep_aspect_ratio=False)    
    return metadata


class ResizeCrop(BaseTransform):

    """Crop the given image and resize it to desired size.

    Args:
        top: Upper pixel coordinate.
        left: Left pixel coordinate.
        height: Height of the cropped image.
        width: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
    """

    def __init__(
        self,
        top,
        left,
        height,
        width,
        size
    ):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.size = size

    def image_transform(self, metadata):
        return resize_crop(
            metadata,
            top=self.top,
            left=self.left,
            height=self.height,
            width=self.width,
            size=self.size
        )


class RandomResizeCrop(BaseRandomTransform):

    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        prob=0.5,
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    @staticmethod
    def get_params(image, scale, ratio):
        area = image.shape[0] * image.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if target_w <= image.shape[1] and target_h <= image.shape[0]:
                i = random.randint(0, image.shape[0] - target_h)
                j = random.randint(0, image.shape[1] - target_w)
                return i, j, target_h, target_w

        # Fallback to central crop
        in_ratio = image.shape[1] / image.shape[0]
        if in_ratio < min(ratio):
            target_w = image.shape[1]
            target_h = int(round(target_w / min(ratio)))
        elif in_ratio > max(ratio):
            target_h = image.shape[0]
            target_w = int(round(target_h * max(ratio)))
        else:  # whole image
            target_w = image.shape[1]
            target_h = image.shape[0]
        i = (image.shape[0] - target_h) // 2
        j = (image.shape[1] - target_w) // 2
        return i, j, target_h, target_w

    def image_transform(self, metadata):
        focus_image = get_focus_image_from_metadata(metadata)
        ret = self.get_params(focus_image, self.scale, self.ratio)
        return resize_crop(
            metadata,
            *ret,
            size=self.size
        )
    