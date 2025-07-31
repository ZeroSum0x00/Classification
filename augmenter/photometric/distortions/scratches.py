import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def scratches(metadata, num_scratches=20, alpha=None):
    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        _, image, _, _, _, _ = extract_metadata(clone_data)
    elif isinstance(metadata, np.ndarray):
        metadata_check = False
        image = copy.deepcopy(metadata)
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")

    if image is None:
        return metadata
        
    alpha = alpha if alpha is not None else .5
    h, w = image.shape[:2]
    min_x, min_y = 0, 0
    max_x, max_y = 2 * w, 2 * h

    scratches = np.zeros((max_y, max_x, 3), np.uint8)
    scratches[:] = 0

    for i in range(0, num_scratches):
        x1 = random.randint(min_x, max_x)
        x2 = random.randint(min_x, max_x)
        y1 = random.randint(min_y, max_y)
        y2 = random.randint(min_y, max_y)

        color = tuple([random.randint(0, 255)] * 3)

        cv2.line(scratches, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)

        # additional scratches for main scratch
        num_additional_scratches = random.randint(1, 4)
        prob_threshold = 0.35
        for j in range(0, num_additional_scratches):
            if random.random() < prob_threshold:
                new_color = random.randint(15, 70)

                param_x1 = random.randint(1, 5)
                param_x2 = random.randint(1, 5)
                param_y1 = random.randint(1, 5)
                param_y2 = random.randint(1, 5)
                cv2.line(
                    scratches,
                    (x1 - param_x1, y1 - param_x2),
                    (x2 - param_y1, y2 - param_y2),
                    (new_color, new_color, new_color),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    top, bottom = h // 2, scratches.shape[0] - (h - h // 2)
    left, right = w // 2, scratches.shape[1] - (w - w // 2)

    scratches = scratches[top:bottom, left:right]
    image = cv2.addWeighted(image[:, :, :3], 1.0, scratches, alpha, 0.0)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Scratches(BaseTransform):
    def __init__(self, num_scratches=20, alpha=None):
        self.num_scratches = num_scratches
        self.alpha = alpha

    def image_transform(self, metadata):
        return scratches(metadata, self.num_scratches, self.alpha)


class RandomScratches(BaseRandomTransform):
    def __init__(self, num_scratches=20, alpha=None, prob=0.5):
        self.num_scratches = num_scratches
        self.alpha = alpha
        self.prob = prob

    def image_transform(self, metadata):
        return scratches(metadata, self.num_scratches, self.alpha)
    