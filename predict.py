import os
import cv2
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


def predict(image, engine_file_config, model_file_config, weight_path):
    engine_config = load_config(engine_file_config)
    data_config   = engine_config['Dataset']
    model_config  = load_config(model_file_config)['Model']
    
    classes, num_classes = get_labels(data_config['data_dir'])
    model = load_model(model_config, weight_path, classes)

    augmentor = data_config["data_augmentation"].get('inference')
    if augmentor and isinstance(augmentor, (tuple, list)):
        augmentor = Augmentor(augment_objects=build_augmenter(augmentor))

    target_size = model_config['input_shape']
    normalizer = Normalizer(data_config['data_normalizer'].get('norm_type', 'divide'),
                            target_size=target_size,
                            mean=data_config['data_normalizer'].get('norm_mean'),
                            std=data_config['data_normalizer'].get('norm_std'),
                            interpolation=data_config['data_normalizer'].get('interpolation', 'BILINEAR'))
    color_space = data_config['data_info'].get('color_space', 'RGB')
    deep_channel = 1 if (len(target_size) > 2 and target_size[-1] > 1) else 0

    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)

        if color_space.lower() != 'bgr':
            image = change_color_space(image, 'bgr' if deep_channel else 'gray', color_space)
            
        if augmentor:
            image = augmentor(image)
        
        image = normalizer(image)

    top1, predict = detect_images([image], model, classes)
    return top1, predict


def predict_folder(data_path, engine_file_config, model_file_config, weight_path, batch_size=32):
    engine_config = load_config(engine_file_config)
    data_config   = engine_config['Dataset']
    model_config  = load_config(model_file_config)['Model']
    
    classes, num_classes = get_labels(data_config['data_dir'])
    model = load_model(model_config, weight_path, classes)
    image_names = [sorted(get_files(os.path.join(data_path, cls), ALLOW_IMAGE_EXTENSIONS, cls)) for cls in classes]
    flattened_files = tuple(item for sublist in image_names for item in sublist)

    augmentor = data_config["data_augmentation"].get('inference')
    if augmentor and isinstance(augmentor, (tuple, list)):
        augmentor = Augmentor(augment_objects=build_augmenter(augmentor))

    target_size = model_config['input_shape']
    normalizer = Normalizer(data_config['data_normalizer'].get('norm_type', 'divide'),
                            target_size=target_size,
                            mean=data_config['data_normalizer'].get('norm_mean'),
                            std=data_config['data_normalizer'].get('norm_std'),
                            interpolation=data_config['data_normalizer'].get('interpolation', 'BILINEAR'))
    color_space = data_config['data_info'].get('color_space', 'RGB')
    deep_channel = 1 if (len(target_size) > 2 and target_size[-1] > 1) else 0
    
    true_count  = 0
    total_count = len(flattened_files)
    for batch_images, batch_labels in tqdm(inference_batch_generator(flattened_files, 
                                                                     data_path, 
                                                                     augmentor, 
                                                                     normalizer, 
                                                                     color_space, 
                                                                     deep_channel, 
                                                                     batch_size), total=(total_count // batch_size) + 1):
        top1s, predicts = detect_images(batch_images, model, classes)

        for i in range(len(batch_labels)):
            if top1s[i][0] == batch_labels[i]:
                true_count += 1

    print(true_count, total_count)
    print(f'Accuracy = {true_count / total_count * 100:.2f}%')


if __name__ == "__main__":
    top1, predict = predict("./datasets/test.jpg",
                            "./configs/test/engine.yaml",
                            "./configs/test/model.yaml",
                            "saved_weights/20250320-130528/weights/best_eval_acc.weights.h5")
    # predict_folder("/media/vbpo-101386/DATA1/Datasets/Classification/full_animals/test",
    #                "./configs/test/engine.yaml",
    #                "./configs/test/model.yaml",
    #                "saved_weights/20250320-130528/weights/best_eval_acc.weights.h5",
    #                batch_size=32)