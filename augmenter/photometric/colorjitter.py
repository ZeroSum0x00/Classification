import random
import collections.abc as collections

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from augmenter.meta_transform import ComposeTransform
from .lighting import RandomAdjustBrightness
from .contrastions import RandomAdjustContrast
from .colors import RandomAdjustHue, RandomAdjustSaturation


class ColorJitter(BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transforms = []

        if (isinstance(brightness, (int, float)) and brightness > 0) or (isinstance(brightness, collections.Iterable) and len(brightness) == 2):
            self.transforms.append(RandomAdjustBrightness(brightness))

        if (isinstance(contrast, (int, float)) and contrast > 0) or (isinstance(contrast, collections.Iterable) and len(contrast) == 2):
            self.transforms.append(RandomAdjustContrast(contrast))
            
        if (isinstance(saturation, (int, float)) and saturation > 0) or (isinstance(saturation, collections.Iterable) and len(saturation) == 2):
            self.transforms.append(RandomAdjustSaturation(saturation))
            
        if (isinstance(hue, (int, float)) and hue > 0) or (isinstance(hue, collections.Iterable) and len(hue) == 2):
            self.transforms.append(RandomAdjustHue(hue))
        
    def image_transform(self, image):
        random.shuffle(self.transforms)
        transform = ComposeTransform(self.transforms)
        return transform(image)


class RandomColorJitter(BaseRandomTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.transforms = []
        self.prob       = prob
        
        if (isinstance(brightness, (int, float)) and brightness > 0) or (isinstance(brightness, collections.Iterable) and len(brightness) == 2):
            self.transforms.append(RandomAdjustBrightness(brightness))

        if (isinstance(contrast, (int, float)) and contrast > 0) or (isinstance(contrast, collections.Iterable) and len(contrast) == 2):
            self.transforms.append(RandomAdjustContrast(contrast))
            
        if (isinstance(saturation, (int, float)) and saturation > 0) or (isinstance(saturation, collections.Iterable) and len(saturation) == 2):
            self.transforms.append(RandomAdjustSaturation(saturation))
            
        if (isinstance(hue, (int, float)) and hue > 0) or (isinstance(hue, collections.Iterable) and len(hue) == 2):
            self.transforms.append(RandomAdjustHue(hue))
        
    def image_transform(self, image):
        random.shuffle(self.transforms)
        transform = ComposeTransform(self.transforms)
        return transform(image)