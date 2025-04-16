import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from utils.logger import logger
from utils.auxiliary_processing import change_color_space


def get_labels(label_object):
    if os.path.isfile(label_object):
        with open(label_object, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
        
    elif os.path.isdir(label_object):
        label_object = f"{label_object}/train/" if os.path.isdir(f"{label_object}/train/") else label_object
        subfolders = [f.path for f in os.scandir(label_object) if f.is_dir() ]
        subfolders = sorted([x.split('/')[-1] for x in subfolders])
        return subfolders, len(subfolders)


def inference_batch_generator(
    images,
    labels,
    augmentor,
    normalizer,
    color_space,
    batch_size,
):
    batch_images = []
    batch_labels = []

    for image, label in zip(images, labels):
        
        if not isinstance(image, np.ndarray):    
            image = cv2.imread(image)
            
            if color_space.lower() != 'bgr':
                image = change_color_space(image, 'BGR', color_space)
        
        if augmentor:
            image = augmentor(image)
        
        image = normalizer(image)

        batch_images.append(image)
        batch_labels.append(label)

        if len(batch_images) == batch_size:
            yield np.array(batch_images), batch_labels
            batch_images, batch_labels = [], []

    if batch_images:
        yield np.array(batch_images), batch_labels


def detect_images(images, model, class_names, top_k=5):
    
    if isinstance(images, str):
        images = [images]

    batch_images = []

    for image in images:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = change_color_space(image, 'BGR', 'RGB')

        batch_images.append(image)

    # print(batch_images)
    batch_images = np.stack(batch_images, axis=0).astype(np.float32)

    # Dự đoán batch ảnh
    predictions = model.predict(batch_images)
    predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()

    if len(class_names) == 2:
        scores = predictions[:, 0]
        top1_results = [(class_names[int(score >= 0.5)], score) for score in scores]
        all_results = [[
            (class_names[0], 1 - score),
            (class_names[1], score)
        ] for score in scores]
    else:
        all_results = decode_predictions(predictions, class_names, top_k)
        top1_results = [pred[0] for pred in all_results]

    return top1_results, all_results


def decode_predictions(preds, class_names, top_k=5):
    top_indices = tf.argsort(preds, axis=-1, direction='DESCENDING')[:, :top_k]
    sorted_preds = np.take_along_axis(preds, top_indices.numpy(), axis=-1)

    results = [
        [(class_names[i], prob) for i, prob in zip(indices, probs)]
        for indices, probs in zip(top_indices.numpy(), sorted_preds)
    ]
    
    return results