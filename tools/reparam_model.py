import os
from utils.post_processing import get_labels
from utils.config_processing import load_config
from predict import load_model
from models import *


if __name__ == "__main__":
    engine_file_config = "saved_weights/20250419-010022/model.yaml"
    model_file_config = "saved_weights/20250419-010022/engine.yaml"
    classes_file = "saved_weights/20250419-010022/classes.names"
    weight_path = "saved_weights/20250419-010022/weights/best_valid_accuracy.weights.h5"

    model_config  = load_config(model_file_config)["Model"]
    engine_config = load_config(engine_file_config)    
    data_config = engine_config["Dataset"]
    classes, num_classes = get_labels(classes_file)

    model = load_model(weight_path, model_config, classes)
    backbone = model.layers[0].layers[0]

    args = {"include_head": True, "inputs": model_config["inputs"], "num_classes": num_classes, "deploy": True}
    structure = RepVGG_B1(**args)
    reparamed_model = repvgg_reparameter(backbone, structure, input_shape=model_config["inputs"], classes=num_classes)
    
    if weight_path.split(".")[-1] == "keras":
        save_path = os.path.join(os.path.dirname(weight_path), "reparamed.keras")
        reparamed_model.save(save_path)
    else:
        save_path = os.path.join(os.path.dirname(weight_path), "reparamed.weights.h5")
        reparamed_model.save_weights(save_path)
        