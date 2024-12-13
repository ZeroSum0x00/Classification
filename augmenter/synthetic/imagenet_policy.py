from augmenter.geometric import *
from augmenter.photometric import *
from augmenter.base_transform import *
from augmenter.meta_transform import *


class ImageNetPolicy(BaseTransform):
  
    def __init__(self):
        self.transform = RandomChoice([
            RandomOrder([
                RandomPosterize(bits=8, prob=0.4), RandomRotation(degrees=50, prob=0.6)
            ]),
            RandomOrder([
                RandomSolarize(threshold=5, prob=0.4), RandomAdjustContrast(contrast_range=9, prob=0.6)
            ]),
            RandomOrder([
                RandomPosterize(bits=7, prob=0.6), RandomPosterize(bits=6, prob=0.6)
            ]),
            RandomOrder([
                RandomRotation(degrees=30, prob=0.2), RandomSolarize(threshold=8, prob=0.6)
            ]),
        ])

    def image_transform(self, image):
        return self.transform(image)