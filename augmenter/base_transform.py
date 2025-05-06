import random
from abc import ABC, abstractmethod


class BaseTransform(ABC):

    @abstractmethod
    def image_transform(self, image):
        raise NotImplemented

    def __call__(self, images):
        if isinstance(images, (tuple, list)):
            images = [self.image_transform(img) for img in images]
        else:
            images = self.image_transform(images)
        return images


class BaseRandomTransform(ABC):
    def __init__(self, prob=0.5):
        self.prob = prob

    @abstractmethod
    def image_transform(self, image):
        raise NotImplemented

    def __call__(self, images):
        if random.random() < self.prob:
            if isinstance(images, (tuple, list)):
                images = [self.image_transform(img) for img in images]
            else:
                images = self.image_transform(images)
        return images
    