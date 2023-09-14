import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from models.architectures.xception import Xception
from models.architectures.vgg import VGG16

from models.classification import CLS
from callbacks import AdvanceWarmUpLearningRate, AccuracyHistory, LossHistory

from data_utils.data_flow import get_train_test_data
from utils.logger import logger
from utils.train_processing import create_folder_weights, train_prepare


def train(data_path,
          data_dst_path,
          color_space,
          data_normalizer,
          data_mean_norm,
          data_std_norm,
          data_augmentation,
          data_type,
          check_data,
          load_memory,
          classes,
          input_shape,
          batch_size,
          init_epoch,
          end_epoch,
          lr_init,
          lr_end,
          weight_type,
          weight_objects,
          show_frequency,
          saved_weight_frequency,
          saved_path,
          training_mode):
          
    if train_prepare(training_mode):
        TRAINING_TIME_PATH = create_folder_weights(saved_path)
        num_classes = len(classes)

        train_generator, val_generator = get_train_test_data(data_zipfile            = data_path, 
                                                             dst_dir                 = data_dst_path,
                                                             classes                 = classes, 
                                                             target_size             = input_shape, 
                                                             batch_size              = batch_size, 
                                                             init_epoch              = init_epoch,
                                                             end_epoch               = end_epoch,
                                                             color_space             = color_space,
                                                             augmentor               = data_augmentation,
                                                             normalizer              = data_normalizer,
                                                             mean_norm               = data_mean_norm,
                                                             std_norm                = data_std_norm,
                                                             data_type               = data_type,
                                                             check_data              = check_data, 
                                                             load_memory             = load_memory)
        
        architecture = VGG16(input_shape=input_shape, classes=num_classes, weights=None)

        model = CLS(architecture, input_shape)
        
        if weight_type and weight_objects:
            if weight_type == "weights":
                model.load_weights(weight_objects)
            elif weight_type == "models":
                model.load_models(weight_objects)

        nbs             = 64
        lr_limit_max    = 1e-1
        lr_limit_min    = 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * lr_init, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * lr_end, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
              
        if num_classes == 2:
            loss = BinaryCrossentropy()
            entropy_metric = BinaryAccuracy()
        else:
            loss = CategoricalCrossentropy()
            entropy_metric = CategoricalAccuracy()

        loss_history = LossHistory(result_path=TRAINING_TIME_PATH)
        
        accuracy_history = AccuracyHistory(entropy_metric, result_path=TRAINING_TIME_PATH, save_best=True)
        
        warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=end_epoch, result_path=TRAINING_TIME_PATH)
        
        callbacks = [accuracy_history, loss_history, warmup_lr]

        optimizer = SGD(learning_rate=Init_lr_fit, momentum=0.9, nesterov=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=[entropy_metric])
        
        model.fit(train_generator,
                  steps_per_epoch     = train_generator.n // batch_size,
                  validation_data     = valid_generator,
                  validation_steps    = valid_generator.n // batch_size,
                  epochs              = end_epoch,
                  initial_epoch       = init_epoch,
                  callbacks           = callbacks)
        
        
if __name__ == '__main__':
    from augmenter import *

    input_shape = (224, 224, 3)
    
    data_path = "/home/vbpo/Desktop/TuNIT/working/Datasets/full_animals"
    data_dst_path = None
    color_space = 'rgb'
    data_normalizer = 'sub_divide'
    # data_mean_norm = [0.485, 0.456, 0.406]
    # data_std_norm = [0.229, 0.224, 0.225]
    data_mean_norm = None
    data_std_norm = None
    data_augmentation = {
                "train": [
                            ResizePadded((224, 224, 3), flexible=True, padding_color=128), 
                            RandomFlip(mode='horizontal'), 
                            RandomRotate(angle_range=20, prob=0.5, padding_color=128),
                            LightIntensityChange(),
                ],
                "valid": [ResizePadded((224, 224, 3), flexible=False, padding_color=128)],
                "test": None
    }
    
    data_type = None
    check_data = False
    load_memory = False

    from utils.post_processing import get_labels
    classes, num_classes = get_labels("./configs/classes.names")
    batch_size = 16
    init_epoch = 0
    end_epoch = 200
    lr_init = 1e-2
    lr_end = lr_init * 0.01
    weight_type = None
    weight_objects = [        
                                    {
                                      'path': './saved_weights/20220926-100327/best_weights_mAP',
                                      'stage': 'full',
                                      'custom_objects': None
                                    }
                                  ]

    show_frequency = 10
    saved_weight_frequency = 100
    saved_path = './saved_weights/'
    training_mode = 'graph'
    
    train(data_path,
          data_dst_path,
          color_space,
          data_normalizer,
          data_mean_norm,
          data_std_norm,
          data_augmentation,
          data_type,
          check_data,
          load_memory,
          classes,
          input_shape,
          batch_size,
          init_epoch,
          end_epoch,
          lr_init,
          lr_end,
          weight_type,
          weight_objects,
          show_frequency,
          saved_weight_frequency,
          saved_path,
          training_mode)
