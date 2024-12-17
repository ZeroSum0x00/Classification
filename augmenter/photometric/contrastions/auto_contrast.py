import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def auto_contrast(image):
    if not is_numpy_image(image):
      raise TypeError('img should be image. Got {}'.format(type(image)))

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = np.min(image).astype(np.float32)
        hi = np.max(image).astype(np.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            im = np.clip(im, 0.0, 255.0)
            return im.astype(np.uint8)

        result = np.where(hi > lo, scale_values(image), image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = np.stack([s1, s2, s3], 2)
    return image


class AutoContrast(BaseTransform):

    def image_transform(self, image):
        return auto_contrast(image)


class RandomAutoContrast(BaseRandomTransform):

    def image_transform(self, image):
        return auto_contrast(image)