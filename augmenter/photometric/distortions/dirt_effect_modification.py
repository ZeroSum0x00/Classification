import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image



def dirt_effect_modification(
    image,
    blur_kernel=(3, 3),
    emboss_kernel_size=None,
    alpha=None,
):
    
    def create_emboss_kernel_top_down(size):
        assert size % 2 == 1, "Kernel must be of an uneven size!"
        k = np.ones((size, size), dtype=np.int32)
        for i in range(size):
            for j in range(size):
                k[i][j] = -1
                if i > (size - 1) / 2:
                    k[i][j] = 1
                if i == (size - 1) / 2:
                    k[i][j] = 0
        return k

    if not is_numpy_image(image):
        raise TypeError("img should be image. Got {}".format(type(image)))
        
    emboss_kernel_size = random.choice([9, 11]) if emboss_kernel_size is None else emboss_kernel_size
    alpha = random.uniform(0.4, 0.7) if alpha is None else alpha

    h, w = image.shape[:2]
    k_size = max(int((h + w) // 300), 3)
    blur_kernel = k_size, k_size

    random_noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    dirt_kernel = create_emboss_kernel_top_down(emboss_kernel_size)
    dirt_colour = cv2.filter2D(random_noise, -1, dirt_kernel)
    gray_dirt = cv2.cvtColor(dirt_colour, cv2.COLOR_BGR2GRAY)
    gray_dirt_3_channels = cv2.cvtColor(gray_dirt, cv2.COLOR_GRAY2BGR)

    blurred_dirt = cv2.blur(gray_dirt_3_channels, blur_kernel)
    final = cv2.addWeighted(image, 1.0, blurred_dirt, alpha, 0.0)
    return final


class DirtEffectModification(BaseTransform):
    def __init__(
        self,
        blur_kernel=(3, 3),
        emboss_kernel_size=None,
        alpha=None,
    ):
        self.blur_kernel = blur_kernel
        self.emboss_kernel_size = emboss_kernel_size
        self.alpha = alpha

    def image_transform(self, image):
        return dirt_effect_modification(
            image=image,
            blur_kernel=self.blur_kernel,
            emboss_kernel_size=self.emboss_kernel_size,
            alpha=self.alpha,
        )

class RandomDirtEffectModification(BaseRandomTransform):
    def __init__(
        self,
        blur_kernel=(3, 3),
        emboss_kernel_size=None,
        alpha=None,
        prob=0.5,
    ):
        self.blur_kernel = blur_kernel
        self.emboss_kernel_size = emboss_kernel_size
        self.alpha = alpha
        self.prob = prob

    def image_transform(self, image):
        return dirt_effect_modification(
            image=image,
            blur_kernel=self.blur_kernel,
            emboss_kernel_size=self.emboss_kernel_size,
            alpha=self.alpha,
        )
    