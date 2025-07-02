import os
import cv2
import numpy as np
import tensorflow as tf
from utils.auxiliary_processing import change_color_space



def get_labels(label_source):
    def process_single_source(source):
        if os.path.isfile(source):
            with open(source, encoding="utf-8") as f:
                class_names = f.readlines()
            return [c.strip() for c in class_names]

        elif os.path.isdir(source):
            if os.path.isfile(os.path.join(source, "train.txt")):
                with open(os.path.join(source, "train.txt"), encoding="utf-8") as f:
                    data = f.readlines()
                classes = {c.strip().split("\t")[1] for c in data}
                return list(classes)
            else:
                train_dir = os.path.join(source, "train")
                label_dir = train_dir if os.path.isdir(train_dir) else source
    
                subfolders = [f.path for f in os.scandir(label_dir) if f.is_dir()]
                return sorted([os.path.basename(x) for x in subfolders])
        else:
            raise ValueError(f"Invalid path: {source}")

    if isinstance(label_source, (list, tuple)):
        all_labels = []
        for src in label_source:
            labels = process_single_source(src)
            all_labels.extend(labels)

        all_labels = sorted(set(all_labels))
        return all_labels, len(all_labels)

    elif isinstance(label_source, str):
        class_names = process_single_source(label_source)
        return class_names, len(class_names)

    else:
        raise TypeError("label_source must be a string, list, or tuple")


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
            
            if color_space.lower() != "bgr":
                image = change_color_space(image, "BGR", color_space)
        
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
            image = change_color_space(image, "BGR", "RGB")

        batch_images.append(image)

    batch_images = np.stack(batch_images, axis=0).astype(np.float32)

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
    top_indices = tf.argsort(preds, axis=-1, direction="DESCENDING")[:, :top_k]
    sorted_preds = np.take_along_axis(preds, top_indices.numpy(), axis=-1)

    results = [
        [(class_names[i], prob) for i, prob in zip(indices, probs)]
        for indices, probs in zip(top_indices.numpy(), sorted_preds)
    ]
    
    return results
