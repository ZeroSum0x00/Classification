import copy
import importlib
from .binary_accuracy import BinaryAccuracy
from .categorical_accuracy import CategoricalAccuracy, TopKCategoricalAccuracy
from .sparse_categorical_accuracy import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy


def build_metrics(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    metrics = []
    if config:
        for cfg in config:
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]
            if not value:
                value = {}
            arch = getattr(mod, name)(**value)
            metrics.append(arch)
    return metrics