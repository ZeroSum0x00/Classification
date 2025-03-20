import os
import copy
from .evaluate_report import Evaluate
from .loss_history import LossHistory
from .metric_history import MetricHistory
from .warmup_lr import AdvanceWarmUpLearningRate
from .train_logger import TrainLogger
from .train_summary import TrainSummary
from tensorflow.keras.callbacks import *
from utils.auxiliary_processing import dynamic_import


def build_callbacks(config, result_path):
    config = copy.deepcopy(config)
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

            arch = dynamic_import(name, globals())(save_path, **value)
            callbacks.append(arch)
    return callbacks
