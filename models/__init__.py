import os
import copy
import inspect
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *

from .train_model import TrainModel
from .classifycation import CLS
from .architectures import *
from .layers import *
from utils.post_processing import get_labels
from utils.auxiliary_processing import dynamic_import
from utils.model_processing import create_layer_instance
from utils.logger import logger


def parse_layer(layer_list):
    parsed = []
    for item in layer_list:
        for layer_name, params in item.items():
            layer_instance = eval(layer_name)
            if layer_instance is None:
                raise ValueError(f"Unknown layer: {layer_name}")
            if params is None:
                parsed.append(layer_instance())
            elif isinstance(params, list):
                nested = parse_layer(params)
                parsed.append(layer_instance(nested))
            elif isinstance(params, dict):
                parsed.append(layer_instance(**params))
            else:
                raise ValueError(f"Invalid parameters for layer {layer_name}")
    return parsed


def get_all_layer(model):
    return [layer.name for layer in model.layers]
    

def is_has_layer(model, layer_name):
    return get_layer_recursive(model, layer_name) is not None


def get_layer_recursive(model, layer_name):
    for layer in getattr(model, "layers", []):
        if layer.name == layer_name:
            return layer

        if hasattr(layer, "layers"):
            found = get_layer_recursive(layer, layer_name)
            if found is not None:
                return found

    return None


def get_model_backbone(model):
    if hasattr(model, "backbone"):
        return model.backbone

    if getattr(model, "layers", None):
        return model.layers[0]

    raise ValueError("Invalid .keras model: cannot find backbone.")

def set_trainable_recursive(layer, trainable):
    layer.trainable = trainable
    for child in getattr(layer, "layers", []):
        set_trainable_recursive(child, trainable)


def freeze_until(freeze_layer, model, verbose=True):
    if isinstance(freeze_layer, str):
        freeze_flag = True
        for layer in model.layers:
            if layer.name == freeze_layer:
                freeze_flag = False
                set_trainable_recursive(layer, True)
            elif freeze_flag:
                set_trainable_recursive(layer, False)
            else:
                set_trainable_recursive(layer, True)

            if verbose:
                print(f"{layer.name} -> trainable: {layer.trainable}")

    elif isinstance(freeze_layer, int):
        if verbose:
            print(f"Total layers: {len(model.layers)}")
            
        total = len(model.layers)
        # Kiểm tra đầu vào có hợp lệ không
        if freeze_layer > total:
            print(f"[WARNING] freeze_layer={freeze_layer} lớn hơn tổng số layer. Tất cả sẽ bị unfreeze.")
                
            freeze_layer = total

        if freeze_layer == -1:
            set_trainable_recursive(model, True)
        elif freeze_layer == 0:
            set_trainable_recursive(model, False)
        else:
            for layer in model.layers[:-freeze_layer]:
                set_trainable_recursive(layer, False)
            for layer in model.layers[-freeze_layer:]:
                set_trainable_recursive(layer, True)
            
        if verbose:
            for layer in model.layers:
                print(f"{layer.name} -> trainable: {layer.trainable}")


# def get_freeze_layer_input():
#     freeze_layer_index = input("Enter freeze layer: ")  # Người dùng nhập vào
#     try:
#         # Nếu nhập được số, trả về kiểu int
#         return int(freeze_layer_index)
#     except ValueError:
#         # Nếu không thể ép kiểu sang số, trả về tên lớp (string)
#         return freeze_layer_index


def remove_last_layer(model):
    trim_count = 1
    if len(model.layers) > 1 and not model.layers[-1].trainable_weights:
        trim_count = 2

    if isinstance(model, tf.keras.Sequential):
        new_layers = model.layers[:-trim_count]
        if not new_layers:
            raise ValueError(f"Cannot remove classifier output from empty head: {model.name}")
        new_model = tf.keras.Sequential(new_layers, name=model.name)
    else:
        new_output = model.layers[-trim_count - 1].output
        new_model = tf.keras.Model(inputs=model.input, outputs=new_output, name=model.name)
    return new_model
    

def cut_model_at_layer(model, layer_name):
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        raise ValueError(f"Layer '{layer_name}' không tồn tại trong mô hình.")

    new_model = tf.keras.Model(inputs=model.input, outputs=target_layer.output, name=f"{model.name}_cut_at_{layer_name}")
    return new_model


def create_model_instance(block, *args, **kwargs):

    if inspect.isclass(block):
        valid_params = inspect.signature(block.__init__).parameters
    elif inspect.isfunction(block):
        valid_params = inspect.signature(block).parameters
    else:
        raise ValueError("block must be either a class or a function")

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return block(*args, **filtered_kwargs)


def normalize_train_strategy(train_strategy):
    train_strategy = (train_strategy or "scratch").lower()
    if train_strategy == "feature-extraction":
        return "feature-extractor"
    return train_strategy


def resolve_classes(classes, is_set_classes=False):
    if is_set_classes:
        return classes, len(classes)

    if not classes:
        raise ValueError("You much pass classes args to get classes and num_classes")

    if isinstance(classes, str):
        return get_labels(classes)

    if isinstance(classes, (list, tuple)):
        if all(isinstance(c, str) and os.path.exists(c) for c in classes):
            return get_labels(classes)
        return list(classes), len(classes)

    return classes, len(classes)


def create_custom_head(custom_head_config, input_shape, name):
    custom_head = Sequential(parse_layer(custom_head_config), name=name)
    custom_head_input = tf.keras.Input(shape=input_shape[1:])
    custom_head(custom_head_input)
    custom_head.trainable = True
    return custom_head


def iter_layers_recursive(layer):
    for child in getattr(layer, "layers", []):
        yield child
        yield from iter_layers_recursive(child)


def sync_trainable_by_name(source_model, target_model):
    trainable_by_name = {
        layer.name: layer.trainable
        for layer in iter_layers_recursive(source_model)
    }

    for layer in iter_layers_recursive(target_model):
        if layer.name in trainable_by_name:
            layer.trainable = trainable_by_name[layer.name]


def freeze_transfer_base(full_model, head_name, freeze_layer):
    base_layers = []
    for layer in full_model.layers:
        if layer.name == head_name:
            break
        base_layers.append(layer)

    if isinstance(freeze_layer, str):
        freeze_flag = True
        for layer in base_layers:
            if layer.name == freeze_layer:
                freeze_flag = False
            set_trainable_recursive(layer, not freeze_flag)
    elif freeze_layer == -1:
        for layer in base_layers:
            set_trainable_recursive(layer, True)
    elif freeze_layer == 0:
        for layer in base_layers:
            set_trainable_recursive(layer, False)
    else:
        total = len(base_layers)
        freeze_layer = min(freeze_layer, total)
        for layer in base_layers[:-freeze_layer]:
            set_trainable_recursive(layer, False)
        for layer in base_layers[-freeze_layer:]:
            set_trainable_recursive(layer, True)

    if head_name in [layer.name for layer in full_model.layers]:
        set_trainable_recursive(full_model.get_layer(head_name), True)


def build_models(trainer_config, model_config):
    trainer_config = copy.deepcopy(trainer_config)
    model_config = copy.deepcopy(model_config)
    train_strategy = normalize_train_strategy(trainer_config["strategy"].get("train_mode", "scratch"))
    inputs = model_config.pop("inputs", [224, 224, 3])
    weight_path = model_config.pop("weight_path", None)
    classes = model_config.pop("classes")
    is_set_classes = model_config.pop("is_set_classes", False)
    model_clip_gradient = trainer_config["advanced"].pop("model_clip_gradient", 5.)
    gradient_accumulation_steps = trainer_config["advanced"].pop("gradient_accumulation_steps", 1)
    sam_rho = trainer_config["advanced"].pop("sam_rho", 0.0)
    use_ema = trainer_config["advanced"].pop("train_with_ema", False)
    compile_jit = trainer_config["advanced"].pop("compile_jit", False)
    
    architecture_config = model_config["Architecture"]
    architecture_name = architecture_config.pop("name")

    classes, num_classes = resolve_classes(classes, is_set_classes)
    is_keras_model = bool(weight_path and weight_path.endswith(".keras"))
    is_h5_weights = bool(weight_path and weight_path.endswith(".h5"))
    use_transfer_strategy = train_strategy in ["fine-tuning", "feature-extractor"]

    if weight_path and not (is_keras_model or is_h5_weights):
        raise ValueError(f"Unsupported weight_path format: {weight_path}")
    
    if is_keras_model:
        loaded_model = tf.keras.models.load_model(weight_path)
        backbone = get_model_backbone(loaded_model)
        logger.info(f"Loaded backbone from saved model: {weight_path}")
    else:
        backbone_config = model_config["Backbone"]
        backbone_config["inputs"] = inputs
        
        if backbone_config.get("weights") and backbone_config.get("weights").lower() == "imagenet":
            backbone_config["num_classes"] = 1000
        else:
            backbone_config["num_classes"] = num_classes

        backbone_name = backbone_config.pop("name")
        backbone = dynamic_import(backbone_name, globals())
        backbone = create_model_instance(backbone, **backbone_config)

    load_weights_after_build = is_h5_weights
    if is_h5_weights and use_transfer_strategy:
        weight_loader_config = copy.deepcopy(architecture_config)
        weight_loader_config["backbone"] = backbone
        weight_loader_config["num_classes"] = num_classes

        weight_loader = dynamic_import(architecture_name, globals())(**weight_loader_config)
        weight_loader.build((None, *inputs))
        weight_loader.load_weights(weight_path, skip_mismatch=True)
        backbone = get_model_backbone(weight_loader)
        load_weights_after_build = False
        logger.info(f"Loaded transfer weights before splitting head: {weight_path}")

    head = get_layer_recursive(backbone, "classifier_head")

    if use_transfer_strategy:
        classifier_head_block = None

        if head is not None:
            dummy_input = tf.keras.Input(shape=head.input_shape[1:])
            head(dummy_input)
            
            classifier_head_block = remove_last_layer(head)
            transfer_head_input = tf.keras.Input(shape=dummy_input.shape[1:])
            transfer_head_output = classifier_head_block(transfer_head_input)

            freeze_model = Model(
                inputs=backbone.input,
                outputs=head.input,
                name=f"{backbone.name}_backbone"
            )

            del dummy_input
            del head
        else:
            backbone_output = backbone.output
            if isinstance(backbone_output, (list, tuple)):
                freeze_model = Model(
                    inputs=backbone.input,
                    outputs=backbone_output[-1],
                    name=f"{backbone.name}_backbone"
                )
            else:
                freeze_model = backbone

        unfreeze_model_config = model_config.get("CustomHead", None)
        if unfreeze_model_config:
            unfreeze_model = Sequential(parse_layer(unfreeze_model_config), name="transfer_classifier_head")
        elif classifier_head_block:
            unfreeze_model = Model(
                inputs=transfer_head_input,
                outputs=transfer_head_output,
                name="transfer_classifier_head"
            )
        else:
            unfreeze_model = None
            
        if train_strategy == "fine-tuning":
            freeze_layer_index = trainer_config["strategy"].get(
                "freeze_layer",
                trainer_config["strategy"].get("model_freeze_layer", -1)
            )
            freeze_until(freeze_layer_index, freeze_model, verbose=False)
        else:
            freeze_model.trainable = False
        
        if unfreeze_model:
            full_model = tf.keras.Model(
                inputs=freeze_model.input,
                outputs=unfreeze_model(freeze_model.output),
                name="full_transfer_model"
            )
            sync_trainable_by_name(freeze_model, full_model)
            if train_strategy == "feature-extractor":
                freeze_transfer_base(full_model, unfreeze_model.name, 0)
            else:
                freeze_transfer_base(full_model, unfreeze_model.name, freeze_layer_index)
            architecture_config["backbone"] = full_model
        else:
            architecture_config["backbone"] = freeze_model
    else:
        if (not is_has_layer(backbone, "classifier_head") or 
            len(backbone.output_shape) != 2 or 
            backbone.output_shape[1] != num_classes):
            
            custom_head_config = model_config.get("CustomHead", None)
            if custom_head_config:
                backbone_output = backbone.output
                if isinstance(backbone_output, (list, tuple)):
                    latted_dim = backbone_output[-1].shape
                else:
                    latted_dim = backbone_output.shape

                head_model = create_custom_head(custom_head_config, latted_dim, name="custom_head")
                architecture_config["custom_head"] = head_model

                del head_model
        architecture_config["backbone"] = backbone
        
    architecture_config["num_classes"] = num_classes
    architecture = dynamic_import(architecture_name, globals())(**architecture_config)

    if load_weights_after_build:
        architecture.build((None, *inputs))
        architecture.load_weights(weight_path, skip_mismatch=True)
        logger.info(f"Loaded architecture weights from: {weight_path}")

    model = TrainModel(
        architecture=architecture,
        classes=classes,
        inputs=inputs,
        teacher_models=None,
        distillation_type="",
        temperature=3,
        alpha=0.1,
        model_clip_gradient=model_clip_gradient,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sam_rho=sam_rho,
        use_ema=use_ema,
        compile_jit=compile_jit,
        name=architecture_name,
    )
    return model
