import os
import cv2
import numpy as np
import tensorflow as tf

from models.architectures.xception import Xception
from models.architectures.vgg import VGG16
from models.classification import CLS
from utils.files import get_files
from utils.post_processing import get_labels, detect_image
from utils.auxiliary_processing import change_color_space
from visualizer.visual_image import visual_image
from configs import general_config as cfg


if __name__ == "__main__":
    input_shape = (224, 224, 3)
    classes, num_classes = get_labels("./configs/classes.names")

    data_path = "/home/vbpo-101386/Desktop/TuNIT/Datasets/Classification/PetImages/test/Dog/"
    image_names = get_files(data_path, extensions=['jpg', 'jpeg', 'png'])

    architecture = Xception(input_shape=input_shape, classes=num_classes, weights=None)
    
    model = CLS(architecture, input_shape)
    
    load_type      = "weights"
    
    weight_objects = [        
                         {
                             'path': './saved_weights/20230916-125836/best_valid_accuracy',
                             'stage': 'full',
                             'custom_objects': None
                         }
                     ]
    
    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)
            
    for idx, name in enumerate(image_names):
        if idx == 10:
            break
        image_path = f"{data_path}{name}"
        image = cv2.imread(image_path)
        image = change_color_space(image, 'bgr', 'rgb')
        top1, predictions = detect_image(image, model, input_shape, classes)
        # visual_image(image, f'prediction: {top1[0]} | {top1[1]*100:0.2f}%')
        print(top1, predictions)
