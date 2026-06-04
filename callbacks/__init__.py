import os
import copy
from .checkpoint_saver import CheckpointSaver
from .evaluate_report import Evaluate
from .loss_history import LossHistory
from .metric_history import MetricHistory
from .warmup_lr import AdvanceWarmUpLearningRate
from .train_logger import TrainLogger
from .train_summary import TrainSummary
from tensorflow.keras.callbacks import *
from utils.auxiliary_processing import dynamic_import


# Callbacks that take result_path / file path as the first positional argument.
RESULT_PATH_CALLBACKS = frozenset({
    "Evaluate",
    "LossHistory",
    "MetricHistory",
    "AdvanceWarmUpLearningRate",
})
FILE_PATH_CALLBACKS = frozenset({"TrainLogger", "TrainSummary"})
# Keras callbacks whose first positional argument is a path (log_dir, filename, ...).
PATH_FIRST_KERAS_CALLBACKS = frozenset({"TensorBoard", "CSVLogger", "ModelCheckpoint"})


def build_callbacks(config, result_path):
    config = copy.deepcopy(config)
    callbacks = []
    if config:
        for cfg in config:
            save_path = result_path
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]

            try:
                extend_path = value.pop("extend_path", None)
                if extend_path is not None:
                    save_path = os.path.join(save_path, extend_path)
            except:
                pass
                
            if not value:
                value = {}

            callback_cls = dynamic_import(name, globals())
            if name in RESULT_PATH_CALLBACKS | FILE_PATH_CALLBACKS | PATH_FIRST_KERAS_CALLBACKS:
                arch = callback_cls(save_path, **value)
            else:
                arch = callback_cls(**value)
            callbacks.append(arch)
    return callbacks
