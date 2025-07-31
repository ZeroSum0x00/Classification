import copy
from .geometric import *
from .photometric import *
from .synthetic import *
from .base_transform import BaseTransform, BaseRandomTransform
from .meta_transform import ComposeTransform, RandomApply, RandomOrder, RandomChoice



def parse_augment_config(aug_list):
    parsed = []
    for item in aug_list:
        for transform_name, params in item.items():
            transform_instance = eval(transform_name)
            if transform_instance is None:
                raise ValueError(f"Unknown transform: {transform_name}")
            if params is None:
                # For transforms without parameters
                parsed.append(transform_instance())
            elif isinstance(params, list):
                # For transforms that take a list (e.g., RandomOrder)
                nested = parse_augment_config(params)
                parsed.append(transform_instance(nested))
            elif isinstance(params, dict):
                # For transforms with parameters
                parsed.append(transform_instance(**params))
            else:
                raise ValueError(f"Invalid parameters for transform {transform_name}")
    return parsed


def build_augmenter(config):
    config = copy.deepcopy(config)

    if config:
        augmenter = parse_augment_config(config)
        
    return augmenter