import os
import copy
import importlib
from .loss_history import LossHistory
from .metric_history import MetricHistory
from .warmup_lr import AdvanceWarmUpLearningRate
from .train_logger import TrainLogger
from .train_summary import TrainSummary
from tensorflow.keras.callbacks import *


def build_callbacks(config, result_path):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    callbacks = []
    if config:
        for cfg in config:
            save_path = result_path
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]
            extend_path = value.pop("extend_path", None)
            if extend_path is not None:
                save_path = os.path.join(save_path, extend_path)
    
            if not value:
                value = {}

            arch = getattr(mod, name)(save_path, **value)
            callbacks.append(arch)
    return callbacks