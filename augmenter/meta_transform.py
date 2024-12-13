import random


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            if isinstance(images, (tuple, list)):
                images = [t(img) for img in images]
            else:
                images = t(images)
        return images


class RandomApply:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob       = prob

    def __call__(self, images):
        if self.prob < random.random():
            return images

        for t in self.transforms:
            if isinstance(images, (tuple, list)):
                images = [t(img) for img in images]
            else:
                images = t(images)
        return images


class RandomOrder:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        order = list(range(len(self.transforms)))
        random.shuffle(order)

        for i in order:
            # print(self.transforms[i].__class__.__name__)
            if isinstance(images, (tuple, list)):
                images = [self.transforms[i](img) for img in images]
            else:
                images = self.transforms[i](images)
        return images


class RandomChoice:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        t = random.choice(self.transforms)
        if isinstance(images, (tuple, list)):
            images = [t(img) for img in images]
        else:
            images = t(images)
        return images