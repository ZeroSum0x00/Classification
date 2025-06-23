import os
import random
import logging
import datetime
import numpy as np
import tensorflow as tf
from .logger import logger



def train_prepare(
    execution_mode="graph",
    vram_usage="full",
    vram_limit_mb=10240,
    mixed_precision_dtype=None,
    num_gpu=0,
    init_seed=-1
):
    try:
        # --- 1. Set random seeds ---
        if init_seed >= 0:
            random.seed(init_seed)
            np.random.seed(init_seed)
            tf.random.set_seed(init_seed)

            # Ensure deterministic ops
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            try:
                tf.config.experimental.enable_op_determinism()
                logging.info("Enabled deterministic ops in TensorFlow.")
            except AttributeError:
                logging.warning("TensorFlow version does not support 'enable_op_determinism'.")

        # --- 2. Set execution mode (CPU/GPU) ---
        if execution_mode.lower() == "cpu":
            tf.config.set_visible_devices([], "GPU")
            logging.info("Training in CPU mode.")
            return True

        # --- 3. Select GPUs ---
        if isinstance(num_gpu, int):
            num_gpu = [num_gpu] if num_gpu >= 0 else []
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, num_gpu))
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            for gpu in gpus:
                if vram_usage.lower() == "limit" and (vram_limit_mb and vram_limit_mb > 0):
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=vram_limit_mb)]
                    )
                    logging.info(f"Limiting GPU memory to {vram_limit_mb}MB.")
                elif vram_usage.lower() == "growth":
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info("Enabled memory growth for GPU.")
                elif vram_usage.lower() == "full":
                    logging.info("Using full GPU memory (default).")
                else:
                    logging.warning(f"Unknown vram_usage mode: {vram_usage}. Using default.")

        # --- 4. Mixed Precision Training ---
        allowed_precisions = ["fp16", "float16", "mixed_float16", "bf16", "bfloat16", "mixed_bfloat16"]
        if mixed_precision_dtype and mixed_precision_dtype.lower() in allowed_precisions:
            from tensorflow.keras import mixed_precision
            if mixed_precision_dtype.lower() in ["fp16", "float16"]:
                policy = "mixed_float16"
            elif mixed_precision_dtype.lower() in ["bf16", "bfloat16"]:
                policy = "mixed_bfloat16"
            else:
                policy = mixed_precision_dtype.lower()
            mixed_precision.set_global_policy(policy)
            logging.info(f"Using mixed precision: {policy}.")

        # --- 5. Execution Mode (Eager / Graph) ---
        if execution_mode.lower() == "eager":
            tf.config.run_functions_eagerly(True)
            logging.info("Training in eager mode.")
        elif execution_mode.lower() == "graph":
            tf.config.run_functions_eagerly(False)
            logging.info("Training in graph mode.")
        else:
            logging.error(f'Invalid execution_mode: {execution_mode}. Use "eager", "graph", or "cpu".')
            return False

        return True

    except Exception as e:
        logging.error(f"[Train prepare error] {e}")
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
        