class Augmentor:
    def __init__(self, augment_objects):
        self.sequence_transform  = augment_objects

    def __call__(self, images):
        if self.sequence_transform:
            for transform in self.sequence_transform:
                images = transform(images)
        return images