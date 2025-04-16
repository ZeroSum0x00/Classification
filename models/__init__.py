# train-test processing model
import copy
from .train_model import TrainModel
from .classifycation import CLS
from .architectures import *
from utils.post_processing import get_labels
from utils.auxiliary_processing import dynamic_import


def build_models(config):
    config = copy.deepcopy(config)
    inputs = config.pop("inputs")
    weight_path = config.pop("weight_path")
    load_weight_type = config.pop("load_weight_type")
    classes = config.pop('classes')

    architecture_config = config['Architecture']
    architecture_name = architecture_config.pop("name")

    backbone_config = config['Backbone']
    backbone_config['inputs'] = inputs
    if classes:
        if isinstance(classes, str):
            classes, num_classes = get_labels(classes)
        else:
            num_classes = len(classes)
        backbone_config['num_classes'] = num_classes

    backbone_name = backbone_config.pop("name")
    backbone = dynamic_import(backbone_name, globals())(**backbone_config)

    architecture_config['backbone'] = backbone
    architecture = dynamic_import(architecture_name, globals())(**architecture_config)

    model = TrainModel(architecture, classes=classes, inputs=inputs, name=architecture_name)
    return model