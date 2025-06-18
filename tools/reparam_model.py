import os
from utils.post_processing import get_labels
from utils.config_processing import load_config
from predict import load_model
from models import *


if __name__ == "__main__":
    engine_file_config = "saved_weights/20250507-155452/engine.yaml"
    model_file_config = "saved_weights/20250507-155452/model.yaml"
    classes_file = "saved_weights/20250507-155452/classes.names"
    weight_path = "saved_weights/20250507-155452/weights/best_valid_accuracy.weights.h5"

    model_config  = load_config(model_file_config)["Model"]
    engine_config = load_config(engine_file_config)    
    data_config = engine_config["Dataset"]
    classes, num_classes = get_labels(classes_file)

    model = load_model(weight_path, model_config, classes)
    if model.name == "CLS":
        backbone = model.layers[0].layers[0]
    else:
        backbone = model
        
    args = {"include_head": True, "weights": None, "inputs": model_config["inputs"], "num_classes": num_classes, "deploy": True}
    structure = RepVGG_A0(**args)
    reparamed_backbone = repvgg_reparameter(backbone, structure, input_shape=model_config["inputs"], classes=num_classes)

    reparamed_model = CLS(backbone=reparamed_backbone)
    
    if weight_path.split(".")[-1] == "keras":
        save_path = os.path.join(os.path.dirname(weight_path), "reparamed.keras")
        reparamed_model.save(save_path)
    else:
        save_path = os.path.join(os.path.dirname(weight_path), "reparamed.weights.h5")
        reparamed_model.build(input_shape=(None, *model_config["inputs"]))
        reparamed_model.save_weights(save_path)
        