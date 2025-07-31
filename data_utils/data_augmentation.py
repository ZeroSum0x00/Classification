class AugmentWrapper:
    def __init__(self, transforms):
        self.transforms  = transforms

    def __call__(self, metadata):
        for t in self.transforms:
            metadata = t(metadata)
        
        return metadata
    