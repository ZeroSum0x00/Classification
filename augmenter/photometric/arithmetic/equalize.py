import numpy as np
from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].astype(np.int32)
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, [0, 256])[0]

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = np.where(np.not_equal(histo, 0))
        nonzero_histo = np.reshape(np.take(histo, nonzero), [-1])
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return np.clip(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = np.where(np.equal(step, 0), im, np.take(build_lut(histo, step), im))

        return result.astype(np.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = np.stack([s1, s2, s3], 2)
    return image


class Equalize(BaseTransform):

    def image_transform(self, image):
        return equalize(image)


class RandomEqualize(BaseRandomTransform):
  
    def image_transform(self, image):
        return equalize(image)