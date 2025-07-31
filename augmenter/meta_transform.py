import random


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, metadata):
        for t in self.transforms:
            metadata = t(metadata)
        return metadata


class RandomApply:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob       = prob

    def __call__(self, metadata):
        if self.prob < random.random():
            return metadata

        for t in self.transforms:
            metadata = t(metadata)
        return metadata


class RandomOrder:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, metadata):
        order = list(range(len(self.transforms)))
        random.shuffle(order)

        for i in order:
            for t in self.transforms:
                metadata = self.transforms[i](metadata)
        return metadata


class RandomChoice:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, metadata):
        t = random.choice(self.transforms)
        metadata = t(metadata)
        return metadata
    