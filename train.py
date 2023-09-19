import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from models import CLS, Xception, VGG16
from callbacks import AdvanceWarmUpLearningRate, LossHistory, MetricHistory
from utils.post_processing import get_labels
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from configs.general_config import *
from utils.logger import logger


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
          dataloader_mode,
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

        train_generator, valid_generator, test_generator = get_train_test_data(data_zipfile            = data_path, 
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
                                                                               load_memory             = load_memory,
                                                                               dataloader_mode         = dataloader_mode)
        
        architecture = Xception(input_shape=input_shape, classes=num_classes, weights=None)

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
            accuracy_metric = BinaryAccuracy(name="accuracy")
            metrics = [accuracy_metric]
        else:
            loss = CategoricalCrossentropy()
            accuracy_metric = CategoricalAccuracy(name="accuracy")
            top_k_accuracy_metric = TopKCategoricalAccuracy(5, name="top-5-accuracy")
            metrics = [accuracy_metric, top_k_accuracy_metric]

        loss_history = LossHistory(result_path=TRAINING_TIME_PATH)
        
        metric_history = MetricHistory(result_path=TRAINING_TIME_PATH, save_best=True)
        
        warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=end_epoch, result_path=TRAINING_TIME_PATH)
        
        callbacks = [metric_history, loss_history, warmup_lr]

        optimizer = SGD(learning_rate=Init_lr_fit, momentum=0.9, nesterov=True)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        if valid_generator is not None:
            model.fit(train_generator,
                      steps_per_epoch     = train_generator.n // batch_size,
                      validation_data     = valid_generator,
                      validation_steps    = valid_generator.n // batch_size,
                      epochs              = end_epoch,
                      initial_epoch       = init_epoch,
                      callbacks           = callbacks)
        else:
            model.fit(train_generator,
                      steps_per_epoch     = train_generator.n // batch_size,
                      epochs              = end_epoch,
                      initial_epoch       = init_epoch,
                      callbacks           = callbacks)
              
        if test_generator is not None:
            model.evaluate(test_generator)


if __name__ == '__main__':
    classes, num_classes = get_labels(DATA_ANNOTATION_PATH)
     
    train(DATA_PATH,
          DATA_DESTINATION_PATH,
          DATA_COLOR_SPACE,
          DATA_NORMALIZER,
          DATA_MEAN_NORMALIZATION,
          DATA_STD_NORMALIZATION,
          DATA_AUGMENTATION,
          DATA_TYPE,
          CHECK_DATA,
          DATA_LOAD_MEMORY,
          TRAIN_DATALOADER_MODE,
          classes,
          INPUT_SHAPE,
          TRAIN_BATCH_SIZE,
          TRAIN_EPOCH_INIT,
          TRAIN_EPOCH_END,
          TRAIN_LR_INIT,
          TRAIN_LR_END,
          TRAIN_WEIGHT_TYPE,
          TRAIN_WEIGHT_OBJECTS,
          TRAIN_RESULT_SHOW_FREQUENCY,
          TRAIN_SAVE_WEIGHT_FREQUENCY,
          TRAIN_SAVED_PATH,
          TRAIN_MODE)
