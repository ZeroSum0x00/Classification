import copy
import tensorflow as tf
from losses.binary_crossentropy import BinaryCrossentropy
from losses.binary_focal_crossentropy import BinaryFocalCrossentropy
from losses.categorical_crossentropy import CategoricalCrossentropy
from losses.categorical_focal_crossentropy import CategoricalFocalCrossentropy
from losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from utils.auxiliary_processing import dynamic_import



def build_losses(config):
    config = copy.deepcopy(config)
    losses = []

    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        if value:
            coeff = value.pop("coeff")
        else:
            coeff = 1
            value = {}
        arch = dynamic_import(name, globals())(**value)
        losses.append({'loss': arch, 'coeff': coeff})
    return losses
