import os
import sys
import logging
import datetime
import tensorflow as tf
from utils.logger import logger
from utils.files import verify_folder

def train_prepare(train_mode):
    try:
        logger.info(f"Setting trainer width {train_mode.lower()} mode")
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
    
    
def create_folder_weights(saved_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    TRAINING_TIME_PATH = saved_dir + current_time
    access_rights = 0o755
    try:
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        logger.info("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return verify_folder(TRAINING_TIME_PATH)
    except: 
        logger.error("Creation of the directory %s failed" % TRAINING_TIME_PATH)
        

def log_training_time(saved_dir):
    f = open(f"{saved_dir}/train_time.log", 'w')
    sys.stdout = f