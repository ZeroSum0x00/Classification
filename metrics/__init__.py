import copy
from .binary_accuracy import BinaryAccuracy
from .categorical_accuracy import CategoricalAccuracy, TopKCategoricalAccuracy
from .sparse_categorical_accuracy import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
from utils.auxiliary_processing import dynamic_import


def build_metrics(config):
    config = copy.deepcopy(config)
    metrics = []
    if config:
        for cfg in config:
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]
            if not value:
                value = {}
            arch = dynamic_import(name, globals())(**value)
            metrics.append(arch)
    return metrics