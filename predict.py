import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from models import build_models
from augmenter import build_augmenter
from data_utils import Augmentor, Normalizer
from utils.files import get_files
from utils.auxiliary_processing import change_color_space
from utils.post_processing import get_labels, detect_images, inference_batch_generator
from utils.constants import ALLOW_IMAGE_EXTENSIONS
from utils.config_processing import load_config


def load_model(model_config, weight_path, classes):
    input_shape = model_config["input_shape"]
    
    if not model_config['classes']:
        model_config['classes'] = classes
    
    model = build_models(model_config)
    model.architecture.build((None, *input_shape))
    model.architecture.load_weights(weight_path)
    return model


def predict(image, model, classes, augmentor, normalizer, color_space="RGB"):
    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

        if color_space.lower() != 'bgr':
            image = change_color_space(image, 'BGR', color_space)
            
    if augmentor:
        image = augmentor(image)
    
    image = normalizer(image)

    top1, predict = detect_images([image], model, classes)
    return top1, predict


def predict_folder(images, model, classes, augmentor, normalizer, color_space="RGB", batch_size=32):    
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

    print(f'Accuracy = {true_count / total_count * 100:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified config files.")
    parser.add_argument(
        "--engine_config", type=str, default="./configs/test/engine.yaml",
        help="Path to the engine configuration YAML file. Default: ./configs/test/engine.yaml"
    )
    parser.add_argument(
        "--model_config", type=str, default="./configs/test/model.yaml",
        help="Path to the model configuration YAML file. Default: ./configs/test/model.yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine_config = load_config(args.engine_config)
    data_config   = engine_config['Dataset']
    model_config  = load_config(args.model_config)['Model']
    weight_path   = "saved_weights/20250320-130528/weights/best_eval_acc.weights.h5"

    data_path = "/media/vbpo-101386/DATA1/Datasets/Classification/full_animals/test"
    files = get_files(data_path, ALLOW_IMAGE_EXTENSIONS)

    classes, num_classes = get_labels(data_config['data_dir'])

    model = load_model(model_config, weight_path, classes)

    augmentor = data_config["data_augmentation"].get('inference')
    if augmentor and isinstance(augmentor, (tuple, list)):
        augmentor = Augmentor(augment_objects=build_augmenter(augmentor))

    normalizer = Normalizer(data_config['data_normalizer'].get('norm_type', 'divide'),
                            target_size=model_config['input_shape'],
                            mean=data_config['data_normalizer'].get('norm_mean'),
                            std=data_config['data_normalizer'].get('norm_std'),
                            interpolation=data_config['data_normalizer'].get('interpolation', 'BILINEAR'))
    color_space = data_config['data_info'].get('color_space', 'RGB')


    # for fi in files:
    #     top1, pred = predict(os.path.join(data_path, fi),
    #                          model=model,
    #                          classes=classes,
    #                          augmentor=augmentor,
    #                          normalizer=normalizer,
    #                          color_space=color_space)
    #     print(top1)
    
    image_names = [sorted(get_files(os.path.join(data_path, cls), ALLOW_IMAGE_EXTENSIONS, cls)) for cls in classes]
    flattened_files = tuple(os.path.join(data_path, item) for sublist in image_names for item in sublist)

    predict_folder(flattened_files,
                   model=model,
                   classes=classes,
                   augmentor=augmentor,
                   normalizer=normalizer,
                   color_space=color_space,
                   batch_size=32)
