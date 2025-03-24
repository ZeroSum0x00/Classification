import os
import sys
import random
import logging
import datetime
import numpy as np
import tensorflow as tf
from utils.logger import logger


def losses_prepare(loss_object):
    loss = loss_object['loss']
    loss.coefficient = loss_object['coeff']
    return loss


import os
import random
import numpy as np
import tensorflow as tf
import logging

def train_prepare(train_mode, vram_usage='limit', num_gpu=0, init_seed=-1):
    try:
        if init_seed >= 0:
            random.seed(init_seed)
            np.random.seed(init_seed)
            tf.random.set_seed(init_seed)
        
        if train_mode == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
            return True
        else:
            if isinstance(num_gpu, int):
                num_gpu = [num_gpu] if num_gpu >= 0 else []
            
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, num_gpu))
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if vram_usage and gpus:
                try:
                    for gpu in gpus:
                        if vram_usage.lower() == "limit":
                            tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)

            logger.info(f"Setting trainer with {train_mode.lower()} mode")
            if train_mode.lower() == 'eager':
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                tf.get_logger().setLevel(logging.ERROR)
                tf.config.run_functions_eagerly(True)
                return True
            elif train_mode.lower() == 'graph':
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                tf.get_logger().setLevel(logging.ERROR)
                tf.config.run_functions_eagerly(False)
                return True
            else:
                logger.error(f"Can't find {train_mode} mode. You only choose 'eager' or 'graph'")
                return False
    except BaseException as e:
        print(e)
        return False

    

def find_max_batch_size(model, max_batch=1024):
    def can_run(batch_size):
        try:
            input_data = tf.random.uniform((batch_size, *model.image_size))
            _ = model(input_data)
            return True
        except tf.errors.ResourceExhaustedError:
            return False

    low, high = 1, max_batch
    best_batch_size = low

    while low <= high:
        mid = (low + high) // 2
        if can_run(mid):
            best_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_batch_size


def create_folder_weights(saved_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    TRAINING_TIME_PATH = saved_dir + current_time
    access_rights = 0o755
    try:
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        logger.info("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return TRAINING_TIME_PATH
    except: 
        logger.error("Creation of the directory %s failed" % TRAINING_TIME_PATH)