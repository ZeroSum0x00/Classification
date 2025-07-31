import random
from abc import ABC, abstractmethod



class BaseTransform(ABC):

    @abstractmethod
    def image_transform(self, metadata):
        raise NotImplemented

    def __call__(self, metadata):
        return self.image_transform(metadata)


class BaseRandomTransform(ABC):
    def __init__(self, prob=0.5):
        self.prob = prob

    @abstractmethod
    def image_transform(self, metadata):
        raise NotImplemented

    def __call__(self, metadata):
        if random.random() < self.prob:
            metadata = self.image_transform(metadata)
        return metadata
    