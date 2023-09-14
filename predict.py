import os
import cv2
import numpy as np
import tensorflow as tf

from models.architectures.xception import Xception
from models.architectures.vgg import VGG16
from models.classification import CLS
from utils.files import get_files
from utils.post_processing import get_labels, detect_image
from visualizer.visual_image import visual_image
from configs import general_config as cfg


if __name__ == "__main__":
    input_shape = (112, 112, 3)
    classes, num_classes = get_labels("./configs/classes.names")
            
    image_path = "/home/vbpo-101386/Desktop/TuNIT/Datasets/Classification/PetImages/test/Dog/"
    images = get_files(image_path, extensions=['jpg', 'jpeg', 'png'])

    architecture = VGG16(input_shape=input_shape, classes=num_classes, weights=None)
    model = CLS(architecture, input_shape)
    
    load_type      = "weights"
    
    weight_objects = [        
                         {
                             'path': './saved_weights/20230914-115856/best_valid_accuracy',
                             'stage': 'full',
                             'custom_objects': None
                         }
                     ]
    
    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)
            
    for idx, name in enumerate(images):
        if idx == 10:
            break
        path = f"{image_path}{name}"
        top1, predictions = detect_image(path, model, input_shape, classes)
        print(top1)
