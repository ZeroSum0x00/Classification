import os
import shutil
import argparse
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from metrics import build_metrics
from callbacks import build_callbacks
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from utils.config_processing import load_config


def train(model_file_config, engine_file_config):
    engine_config = load_config(engine_file_config)
    model_config  = load_config(model_file_config)
    data_config  = engine_config['Dataset']
    model_config = model_config['Model']
    train_config = engine_config['Train']
    loss_config = engine_config['Losses']
    optimizer_config = engine_config['Optimizer']
    metric_config = engine_config['Metrics']
    callbacks_config = engine_config['Callbacks']

    if train_prepare(train_config['mode'], num_gpu="0", init_seed=train_config['random_seed']):
        TRAINING_TIME_PATH = create_folder_weights(train_config['save_weight_path'])
        shutil.copy(model_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(model_file_config)))
        shutil.copy(engine_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(engine_file_config)))
        
        model = build_models(model_config)
        train_generator, valid_generator, test_generator = get_train_test_data(data_dirs       = data_config['data_dir'],
                                                                               classes         = model.classes,
                                                                               target_size     = model_config['input_shape'],
                                                                               batch_size      = train_config['batch_size'],
                                                                               color_space     = data_config['data_info']['color_space'],
                                                                               augmentor       = data_config['data_augmentation'],
                                                                               normalizer      = data_config['data_normalizer']['norm_type'],
                                                                               mean_norm       = data_config['data_normalizer']['norm_mean'],
                                                                               std_norm        = data_config['data_normalizer']['norm_std'],
                                                                               data_type       = data_config['data_info']['data_type'],
                                                                               check_data      = data_config['data_info']['check_data'],
                                                                               load_memory     = data_config['data_info']['load_memory'],
                                                                               dataloader_mode = data_config['data_loader_mode'])

        losses    = build_losses(loss_config)
        optimizer = build_optimizer(optimizer_config)
        metrics   = build_metrics(metric_config)
        callbacks = build_callbacks(callbacks_config, TRAINING_TIME_PATH)

        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

        if valid_generator is not None:
            model.fit(train_generator,
                      validation_data = valid_generator,
                      epochs          = train_config['epoch']['end'],
                      initial_epoch   = train_config['epoch']['start'],
                      callbacks       = callbacks)
        else:
            model.fit(train_generator,
                      epochs        = train_config['epoch']['end'],
                      initial_epoch = train_config['epoch']['start'],
                      callbacks     = callbacks)

        if test_generator is not None:
            model.evaluate(test_generator)

        model.save_weights(TRAINING_TIME_PATH + 'weights/last_weights', save_format=train_config['save_weight_type'])

if __name__ == '__main__':
    model_file_config = "./configs/models/convolution-base/vgg.yaml"
    engine_file_config = "./configs/engine/train_supervised.yaml"
    train(model_file_config, engine_file_config)