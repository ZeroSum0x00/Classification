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
        with open(label_file, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
    elif os.path.isdir(label_object):
        label_object = f"{label_object}/train/"
        subfolders = [ f.path for f in os.scandir(label_object) if f.is_dir() ]
        subfolders = sorted([x.split('/')[-1] for x in subfolders])
        return subfolders, len(subfolders)


def resize_image(image, target_size, letterbox_image):
    if len(image.shape) > 2:
        h, w, _    = image.shape
    else:
        h, w = image.shape
    if len(target_size) > 2:
        ih, iw, _  = target_size
    else:
        ih, iw = target_size
    if letterbox_image:
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_resized = cv2.resize(image, (nw, nh))
        if len(image.shape) > 2:
            image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=image.dtype)
            image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        else:
            image_paded = np.full(shape=[ih, iw], fill_value=128.0, dtype=image.dtype)
            image_paded[dh:nh+dh, dw:nw+dw] = image_resized
        return image_paded
    else:
        image = cv2.resize(image, (iw, ih))
        return image


def preprocess_input(image):
    image = image / 127.5 - 1
    return image


def decode_predictions(preds, class_names, top_k=5):
    results = []
    for pred in preds:
        top_indices = tf.argsort(pred)[-top_k:][::-1]
        result = [(class_names[i], pred[i].numpy()) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.extend(result)
    return results


def detect_image(image, model, target_shape, class_names, top_k=5):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = change_color_space(image, 'bgr', 'rgb')
    image_data  = resize_image(image, target_shape, letterbox_image=True)
    image_data  = preprocess_input(image_data.astype(np.float32))
    image_data  = np.expand_dims(image_data, axis=0)
    predictions = model.predict(image_data)
    if len(class_names) == 2:
        score = predictions[0][0].numpy()
        if score < 0.5:
            correct_label_idx = 0
            imprecise_label_idx = 1
        else:
            correct_label_idx = 1
            imprecise_label_idx = 0
            
        top1 = (class_names[correct_label_idx], (1 - score) if correct_label_idx == 0 else score)
        predictions = [(top1), (class_names[imprecise_label_idx], (1 - score) if correct_label_idx == 1 else score)]
    else:
        predictions = decode_predictions(predictions, class_names, top_k=5)
        top1 = predictions[0]
    
    # visual_image([image], [f'{top1[0]}: {top1[1]}'])
    return top1, predictions
