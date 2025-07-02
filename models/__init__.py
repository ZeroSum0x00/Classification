# train-test processing model
import gc
import copy
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

def freeze_until(freeze_layer, model, verbose=True):
    if isinstance(freeze_layer, str):
        freeze_flag = True
        for layer in model.layers:
            if layer.name == freeze_layer:
                freeze_flag = False
                layer.trainable = True
            elif freeze_flag:
                layer.trainable = False
            else:
                layer.trainable = True

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
            model.trainable = True
        elif freeze_layer == 0:
            model.trainable = False
        else:
            for layer in model.layers[:-freeze_layer]:
                layer.trainable = False
            for layer in model.layers[-freeze_layer:]:
                layer.trainable = True
            
        if verbose:
            for layer in model.layers:
                print(f"{layer.name} -> trainable: {layer.trainable}")

def get_freeze_layer_input():
    freeze_layer_index = input("Enter freeze layer: ")  # Người dùng nhập vào
    try:
        # Nếu nhập được số, trả về kiểu int
        return int(freeze_layer_index)
    except ValueError:
        # Nếu không thể ép kiểu sang số, trả về tên lớp (string)
        return freeze_layer_index

def remove_last_layer(model):
    if isinstance(model, tf.keras.Sequential):
        new_model = tf.keras.Sequential(model.layers[:-2], name=model.name)
    else:
        new_output = model.layers[-2].output
        new_model = tf.keras.Model(inputs=model.input, outputs=new_output, name=model.name)
    return new_model
    

def build_models(trainer_config, model_config):
    trainer_config = copy.deepcopy(trainer_config)
    model_config = copy.deepcopy(model_config)
    train_strategy = trainer_config.get("train_strategy", "scratch")
    inputs = model_config.pop("inputs", [224, 224, 3])
    weight_path = model_config.pop("weight_path", None)
    classes = model_config.pop("classes")
    model_clip_gradient = trainer_config.pop("model_clip_gradient", 5.)
    gradient_accumulation_steps = trainer_config.pop("gradient_accumulation_steps", 1)
    sam_rho = trainer_config.pop("sam_rho", 0.0)
    use_ema = trainer_config.pop("train_with_ema", False)
    compile_mode = trainer_config.pop("compile_mode")
    
    architecture_config = model_config["Architecture"]
    architecture_name = architecture_config.pop("name")
    
    if classes:
        if isinstance(classes, (str, list, tuple)):
            classes, num_classes = get_labels(classes)
        else:
            num_classes = len(classes)

    if weight_path and weight_path.endswith(".keras"):
        backbone = tf.keras.models.load_model(weight_path).layers[0]
    else:
        backbone_config = model_config["Backbone"]
        backbone_config["inputs"] = inputs
        
        if backbone_config.get("weights") and backbone_config.get("weights").lower() == "imagenet":
            backbone_config["num_classes"] = 1000
        else:
            backbone_config["num_classes"] = num_classes

        backbone_name = backbone_config.pop("name")
        backbone = dynamic_import(backbone_name, globals())(**backbone_config)

    if train_strategy.lower() in ["fine-tuning", "feature-extractor"]:
        classifier_head_layer = backbone.get_layer("classifier_head")
        
        dummy_input = tf.keras.Input(shape=classifier_head_layer.input_shape[1:])
        _ = classifier_head_layer(dummy_input)

        classifier_head =  remove_last_layer(classifier_head_layer)
        _ = classifier_head(dummy_input)

        freeze_model = Model(
            inputs=backbone.input,
            outputs=classifier_head_layer.input,
            name=f"{backbone.name}_backbone"
        )
        if train_strategy.lower() == "fine-tuning":
            freeze_model.summary()
            freeze_layer_index = get_freeze_layer_input()
            freeze_until(freeze_layer_index, freeze_model, verbose=False)
        else:
            freeze_model.trainable = True

        unfreeze_model_config = model_config.get("CustomHead", None)
        if unfreeze_model_config:
            unfreeze_model = Sequential(parse_layer(unfreeze_model_config), name="transfer_classifier_head")
        else:
            unfreeze_model = Model(
                inputs=classifier_head.input,
                outputs=classifier_head.output,
                name="transfer_classifier_head"
            )

        unfreeze_model(classifier_head_layer.input)

        if len(unfreeze_model.output_shape) != 2 or unfreeze_model.output_shape[1] != num_classes:
            unfreeze_model = Sequential([
                unfreeze_model,
                Dense(
                    units=1 if num_classes == 2 else num_classes,
                    activation="sigmoid" if num_classes == 2 else "softmax",
                )
            ], name="transfer_classifier_head")
            
        unfreeze_model.trainable = True

        del dummy_input
        del backbone
        del classifier_head_layer
        del classifier_head
        K.clear_session()
        gc.collect()
        
        architecture_config["backbone"] = freeze_model
        architecture_config["custom_head"] = unfreeze_model
    else:
        architecture_config["backbone"] = backbone

    architecture = dynamic_import(architecture_name, globals())(**architecture_config)

    if weight_path and weight_path.endswith(".h5"):
        architecture.build((None, *inputs))
        architecture.load_weights(weight_path)
        print("CLS load weights")

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
        compile_mode=compile_mode,
        name=architecture_name,
    )
    return model
