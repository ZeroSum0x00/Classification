import copy
import importlib
import tensorflow as tf
from tensorflow.keras.optimizers import *


def build_optimizer(config):
    config = copy.deepcopy(config)
    name = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, name)(**config)
    return arch