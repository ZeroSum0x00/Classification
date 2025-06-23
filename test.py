import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import build_models
from augmenter import build_augmenter
from data_utils import Augmentor, Normalizer
from utils.files import get_files
from utils.auxiliary_processing import change_color_space
from utils.post_processing import get_labels, detect_images, inference_batch_generator
from utils.constants import ALLOW_IMAGE_EXTENSIONS
from utils.config_processing import load_config



def load_model(weight_path, model_config=None, classes=None):
    if weight_path.endswith(".keras"):
        model = tf.keras.models.load_model(weight_path)
    elif weight_path.endswith(".h5"):
        if not model_config:
            raise ValueError("When using an H5 file, you must provide a `model_config`.")

        if not classes:
            raise ValueError("When using an H5 file, you must specify the number of `classes`.")

        if not model_config.get("classes"):
            model_config["classes"] = classes

        model = build_models(model_config)
        model.load_weights(weight_path)
    else:
        raise ValueError("Invalid file format. Please provide a `.keras` or `.h5` file.")

    return model

    

def predict(
    image,
    model,
    classes,
    augmentor,
    normalizer,
    color_space="RGB",
):
    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

        if color_space.lower() != "bgr":
            image = change_color_space(image, "BGR", color_space)
            
    if augmentor:
        image = augmentor(image)
    
    image = normalizer(image)

    top1, predict = detect_images([image], model, classes)
    return top1, predict


def predict_folder(
    images,
    model,
    classes,
    augmentor,
    normalizer,
    color_space="RGB",
    batch_size=32,
):
    true_count  = 0
    total_count = len(images)

    labels = [os.path.basename(os.path.dirname(img)) for img in images]
    for batch_images, batch_labels in tqdm(inference_batch_generator(images,
                                                                     labels,
                                                                     augmentor,
                                                                     normalizer,
                                                                     color_space,
                                                                     batch_size), total=(total_count // batch_size) + 1):
        top1s, predicts = detect_images(batch_images, model, classes)

        for i in range(len(batch_labels)):
            if top1s[i][0] == batch_labels[i]:
                true_count += 1

    print(f"Accuracy = {true_count / total_count * 100:.2f}%")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a model with specified config files.")
#     parser.add_argument(
#         "--engine_config", type=str, default="./configs/test/engine.yaml",
#         help="Path to the engine configuration YAML file. Default: ./configs/test/engine.yaml"
#     )
#     parser.add_argument(
#         "--model_config", type=str, default="./configs/test/model.yaml",
#         help="Path to the model configuration YAML file. Default: ./configs/test/model.yaml"
#     )
#     parser.add_argument(
#         "--classes", type=str, default=None,
#         help="Path to the define classes file. Default: ./configs/test/classes.names"
#     )
#     parser.add_argument(
#         "--weight_path", type=str, default=None,
#         help="Path to the pre-trained weight file (optional). Default: None"
#     )
#     parser.add_argument(
#         "--data_path", type=str, required=True,
#         help="Path to the dataset directory. This argument is required."
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=32,
#         help="Path to the dataset directory. This argument is required."
#     )
#     return parser.parse_args()


if __name__ == "__main__":
    data_path = "/mnt/data_disk/Datasets/Classification/iwaki/validation"
    engine_config = "saved_weights/20250508-081542/engine.yaml"
    model_config = "saved_weights/20250508-081542/model.yaml"
    weight_path = "saved_weights/20250508-081542/weights/best_valid_accuracy.weights.h5"
    class_file = "saved_weights/20250508-081542/classes.names"
    batch_size = 8
    
    engine_config = load_config(engine_config)
    data_config = engine_config["Dataset"]
    
    model_config  = load_config(model_config)["Model"]

    classes, num_classes = get_labels(class_file)

    model = load_model(weight_path, model_config, classes)

    augmentor = data_config["data_augmentation"].get("inference")
    if augmentor and isinstance(augmentor, (tuple, list)):
        augmentor = Augmentor(augment_objects=build_augmenter(augmentor))

    normalizer = Normalizer(
        data_config["data_normalizer"].get("norm_type", "divide"),
        target_size=model_config["inputs"],
        mean=data_config["data_normalizer"].get("norm_mean"),
        std=data_config["data_normalizer"].get("norm_std"),
        interpolation=data_config["data_normalizer"].get("interpolation", "BILINEAR"),
    )
    color_space = data_config["data_info"].get("color_space", "BGR")

    # files = get_files(args.data_path, ALLOW_IMAGE_EXTENSIONS)
    # for fi in files:
    #     top1, pred = predict(
    #         os.path.join(args.data_path, fi),
    #         model=model,
    #         classes=classes,
    #         augmentor=augmentor,
    #         normalizer=normalizer,
    #         color_space=color_space,
    #     )
    #     print(top1)
    
    image_names = [sorted(get_files(os.path.join(data_path, cls), ALLOW_IMAGE_EXTENSIONS, cls)) for cls in classes]
    flattened_files = tuple(os.path.join(data_path, item) for sublist in image_names for item in sublist)

    predict_folder(
        flattened_files,
        model=model,
        classes=classes,
        augmentor=augmentor,
        normalizer=normalizer,
        color_space=color_space,
        batch_size=batch_size,
    )
    