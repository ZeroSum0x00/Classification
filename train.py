import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from metrics import build_metrics
from callbacks import build_callbacks
from data_utils import get_train_test_data, TFDataPipeline
from utils.train_processing import create_folder_weights, find_max_batch_size, train_prepare
from utils.config_processing import load_config


def train(engine_file_config, model_file_config):
    engine_config = load_config(engine_file_config)
    model_config  = load_config(model_file_config)
    data_config  = engine_config['Dataset']
    model_config = model_config['Model']
    train_config = engine_config['Train']
    loss_config = engine_config['Losses']
    optimizer_config = engine_config['Optimizer']
    metric_config = engine_config['Metrics']
    callbacks_config = engine_config['Callbacks']

    if train_prepare(train_config.get('mode', 'graph'),
                     train_config.get('vram_usage', 'limit'),
                     num_gpu="0",
                     init_seed=train_config['random_seed']):
        TRAINING_TIME_PATH = create_folder_weights(train_config['save_weight_path'])
        shutil.copy(model_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(model_file_config)))
        shutil.copy(engine_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(engine_file_config)))
        
        if not model_config['classes']:
            model_config['classes'] = data_config['data_dir']
        
        model = build_models(model_config)
        batch_size = find_max_batch_size(model) if train_config['batch_size'] == -1 else train_config['batch_size']
        train_generator, valid_generator, test_generator = get_train_test_data(data_dirs       = data_config['data_dir'],
                                                                               classes         = model.classes,
                                                                               target_size     = model_config['input_shape'],
                                                                               batch_size      = batch_size,
                                                                               color_space     = data_config['data_info'].get('color_space', 'RGB'),
                                                                               augmentor       = data_config['data_augmentation'],
                                                                               normalizer      = data_config['data_normalizer'].get('norm_type', 'divide'),
                                                                               mean_norm       = data_config['data_normalizer'].get('norm_mean'),
                                                                               std_norm        = data_config['data_normalizer'].get('norm_std'),
                                                                               data_type       = data_config['data_info']['data_type'],
                                                                               check_data      = data_config['data_info'].get('check_data', False),
                                                                               load_memory     = data_config['data_info'].get('load_memory', False),
                                                                               dataloader_mode = data_config.get('dataloader_mode', 'tf'),
                                                                               get_data_mode   = data_config.get('get_data_mode', 2),
                                                                               num_workers     = train_config['num_workers'])
        
        train_step = int(np.ceil(train_generator.N / batch_size))
        train_generator = train_generator.get_dataset() if isinstance(train_generator, TFDataPipeline) else train_generator

        
        losses    = build_losses(loss_config)
        optimizer = build_optimizer(optimizer_config)
        metrics   = build_metrics(metric_config)
        callbacks = build_callbacks(callbacks_config, TRAINING_TIME_PATH)

        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        
        if valid_generator:
            valid_step = int(np.ceil(valid_generator.N / batch_size))
            valid_generator = valid_generator.get_dataset() if isinstance(valid_generator, TFDataPipeline) else valid_generator
            
            model.fit(train_generator,
                      steps_per_epoch  = train_step,
                      validation_data  = valid_generator,
                      validation_steps = valid_step,
                      epochs           = train_config['epoch']['end'],
                      initial_epoch    = train_config['epoch'].get('start', 0),
                      callbacks        = callbacks)
        else:
            model.fit(train_generator,
                      steps_per_epoch  = train_step,
                      epochs        = train_config['epoch']['end'],
                      initial_epoch = train_config['epoch'].get('start', 0),
                      callbacks     = callbacks)

        if test_generator:
            test_generator  = test_generator.get_dataset() if isinstance(test_generator, TFDataPipeline) else test_generator

            model.evaluate(test_generator)

        model.save_weights(os.path.join(TRAINING_TIME_PATH, 'weights', 'last_weights.weights.h5'))

if __name__ == '__main__':
    engine_file_config = "./configs/test/engine.yaml"
    model_file_config  = "./configs/test/model.yaml"
    train(engine_file_config, model_file_config)