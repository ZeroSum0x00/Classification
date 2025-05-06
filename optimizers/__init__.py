import copy
import tensorflow as tf
from tensorflow.keras.optimizers import *
from utils.auxiliary_processing import dynamic_import


def build_optimizer(config):
    config = copy.deepcopy(config)
    name = config.pop("name")
    arch = dynamic_import(name, globals())(**config)
    return arch
