import copy
import importlib
import tensorflow as tf
from losses.binary_crossentropy import BinaryCrossentropy
from losses.categorical_crossentropy import CategoricalCrossentropy
from losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropy


def build_losses(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    losses = []

    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        if value:
            coeff = value.pop("coeff")
        else:
            coeff = 1
            value = {}
        arch = getattr(mod, name)(**value)
        losses.append({'loss': arch, 'coeff': coeff})
    return losses