from .auxiliary_processing import random_to_negative



def enhance_level_to_arg(level):
    return (level / _MAX_LEVEL) * 1.8 + 0.1  # range [0.1, 1.9]


def rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.0
    return level


def posterize_level_to_arg(level):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 4)


def solarize_level_to_arg(level):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return int((level / _MAX_LEVEL) * 256)


def solarize_add_level_to_arg(level):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110)


def shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = random_to_negative(level)
    return level


def translate_level_to_arg(level, translate_const=1.):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = random_to_negative(level)
    return level


_MAX_LEVEL = 10
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
