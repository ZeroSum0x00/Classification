import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def smudges(metadata, number_smudges=None):

    def add_smudge(img):
        h, w, _ = img.shape
        kernel_coef = random.randint(60, 90)
        ks = int((h * 1.25 + w) / kernel_coef)
        line_thickness_coef = random.randint(60, 80)
        line_thickness = int((1.15 * w + 2 * h) // line_thickness_coef)
        blur_kernel_size = (ks, ks)
        smudges = np.zeros((h, w, 3), np.uint8)
        point_height = random.randint(int(h / 10), int(9 * h / 10))
        p1 = (0, point_height)
        p2 = (w, point_height + random.randint(0, h // 25) * random.choice([-1, 1]))
        p1_b = (0, p1[1])
        p2_b = (w, p2[1])

        color = tuple([random.randint(100, 255)] * 3)
        cv2.line(smudges, p1, p2, color, line_thickness)
        cv2.line(smudges, p1_b, p2_b, color, 1)
        blurred_smudges = cv2.blur(smudges, blur_kernel_size)
        opacity = random.uniform(0.18, 0.25)

        final_image = cv2.addWeighted(img, 1.0, blurred_smudges, opacity, 0.0)
        return final_image

    def transpose(img):
        return np.transpose(img, (1, 0, 2)) if len(img.shape) > 2 else np.transpose(img, (1, 0))

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

    number_smudges = number_smudges if number_smudges else random.randint(1, 6)

    flag = random.getrandbits(1)
    image = image if flag else transpose(image)

    for _ in range(number_smudges):
        image = add_smudge(image)

    image = image if flag else transpose(image)

    if metadata_check:
        clone_data["image"] = image
        return clone_data
    else:
        return image


class Smudges(BaseTransform):
    def __init__(self, number_smudges=None):
        self.number_smudges = number_smudges

    def image_transform(self, metadata):
        return smudges(metadata, self.number_smudges)


class RandomSmudges(BaseRandomTransform):
    def __init__(self, number_smudges=None, prob=0.5):
        self.number_smudges = number_smudges
        self.prob = prob

    def image_transform(self, metadata):
        return smudges(metadata, self.number_smudges)
