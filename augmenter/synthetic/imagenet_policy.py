import random
import importlib
from augmenter.geometric import *
from augmenter.photometric import *
from augmenter.base_transform import *
from augmenter.meta_transform import *
from utils.level_to_arg import *


LEVEL_TO_ARG = {
    "RandomAutoContrast": lambda level: (),
    "RandomEqualize": lambda level: (),
    "RandomInversion": lambda level: (),
    "RandomErasing": lambda level: (),
    "RandomRotation": rotate_level_to_arg,
    "RandomPosterize": posterize_level_to_arg,
    "RandomSolarize": solarize_level_to_arg,
    "RandomSolarizeAdd": solarize_add_level_to_arg,
    "RandomColor": enhance_level_to_arg,
    "RandomContrast": enhance_level_to_arg,
    "RandomBrightness": enhance_level_to_arg,
    "RandomSharpness": enhance_level_to_arg,
    "RandomShearX": shear_level_to_arg,
    "RandomShearY": shear_level_to_arg,
    "TranslateX": translate_level_to_arg,
    "TranslateY": translate_level_to_arg,
    "RandomTranslateX": translate_level_to_arg,
    "RandomTranslateY": translate_level_to_arg,
}

class ImageNetPolicy(BaseTransform):

    def __init__(self, policy_mode="v0"):
        self.mod = importlib.import_module(__name__)

        self.available_policies = {
            "v0": self.policy_v0(),
        }
        self.policies = self.available_policies[policy_mode]
        self.translate_const = 0.8

    def image_transform(self, image):
        t = random.choice(self.policies)
        list_aug = []
        for sub_policy in t:
            name, prob, level = sub_policy
            func_lv2arg = LEVEL_TO_ARG[name]
            if name in ["TranslateX", "TranslateY", "RandomTranslateX", "RandomTranslateY"]:
                level = func_lv2arg(level, translate_const=self.translate_const)
            else:
                level = func_lv2arg(level)

            try:
                if not (isinstance(level, (tuple, list)) and level) and not isinstance(level, (int, float)):
                    arch = getattr(self.mod, name)(prob=prob)
                else:
                    arch = getattr(self.mod, name)(level, prob=prob)
            except:
                if not (isinstance(level, (tuple, list)) and level) and not isinstance(level, (int, float)):
                    arch = getattr(self.mod, name)()
                else:
                    arch = getattr(self.mod, name)(level)
            list_aug.append(arch)
        return ComposeTransform(list_aug)(image)
    
    @staticmethod
    def policy_v0():
        policy = [
            [("RandomColor", 0.4, 9), ("RandomEqualize", 0.6, 3)],
            [("RandomSolarize", 0.8, 3), ("RandomEqualize", 0.4, 7)],
            [("RandomSolarize", 0.4, 2), ("RandomSolarize", 0.6, 2)],
            [("RandomColor", 0.2, 0), ("RandomEqualize", 0.8, 8)],
            [("RandomEqualize", 0.4, 8), ("RandomSolarizeAdd", 0.8, 3)],
            [("RandomColor", 0.6, 1), ("RandomEqualize", 1.0, 2)],
            [("RandomColor", 0.4, 7), ("RandomEqualize", 0.6, 0)],
            [("RandomPosterize", 0.4, 6), ("RandomAutoContrast", 0.4, 7)],
            [("RandomSolarize", 0.6, 8), ("RandomColor", 0.6, 9)],
            [("RandomEqualize", 0.8, 4), ("RandomEqualize", 0.0, 8)],
            [("RandomEqualize", 1.0, 4), ("RandomAutoContrast", 0.6, 2)],
            [("RandomPosterize", 0.8, 2), ("RandomSolarize", 0.6, 10)],
            [("RandomSolarize", 0.6, 8), ("RandomEqualize", 0.6, 1)],
            [("RandomEqualize", 0.8, 1), ("RandomShearY", 0.8, 4)],
            [("RandomColor", 0.4, 1), ("RandomRotation", 0.6, 8)],
            [("RandomShearX", 0.2, 9), ("RandomRotation", 0.6, 8)],
            [("RandomInversion", 0.4, 9), ("RandomRotation", 0.6, 0)],
            [("RandomEqualize", 1.0, 9), ("RandomShearY", 0.6, 3)],
            [("RandomSolarize", 0.2, 4), ("RandomRotation", 0.8, 9)],
            [("RandomRotation", 1.0, 7), ("RandomTranslateY", 0.8, 9)],
            [("RandomShearX", 0.0, 0), ("RandomSolarize", 0.8, 4)],
            [("RandomShearY", 0.8, 0), ("RandomColor", 0.6, 4)],
            [("RandomColor", 1.0, 0), ("RandomRotation", 0.6, 2)],
            [("RandomShearY", 0.4, 7), ("RandomSolarizeAdd", 0.6, 7)],
            [("RandomColor", 0.8, 6), ("RandomRotation", 0.4, 5)],
        ]
        return policy
    
    @staticmethod
    def policy_simple():
        policy = [
            [("RandomColor", 0.4, 9), ("RandomEqualize", 0.6, 3)],
            [("RandomSolarize", 0.8, 3), ("RandomEqualize", 0.4, 7)],
            [("RandomSolarize", 0.4, 2), ("RandomSolarize", 0.6, 2)],
            [("RandomColor", 0.2, 0), ("RandomEqualize", 0.8, 8)],
            [("RandomEqualize", 0.4, 8), ("RandomSolarizeAdd", 0.8, 3)],
            [("RandomColor", 0.6, 1), ("RandomEqualize", 1.0, 2)],
            [("RandomColor", 0.4, 7), ("RandomEqualize", 0.6, 0)],
            [("RandomPosterize", 0.4, 6), ("RandomAutoContrast", 0.4, 7)],
            [("RandomSolarize", 0.6, 8), ("RandomColor", 0.6, 9)],
            [("RandomEqualize", 0.8, 4), ("RandomEqualize", 0.0, 8)],
            [("RandomEqualize", 1.0, 4), ("RandomAutoContrast", 0.6, 2)],
            [("RandomPosterize", 0.8, 2), ("RandomSolarize", 0.6, 10)],
            [("RandomSolarize", 0.6, 8), ("RandomEqualize", 0.6, 1)],
        ]
        return policy
