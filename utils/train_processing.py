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
                logger.info("Enabled deterministic ops in TensorFlow.")
            except AttributeError:
                logger.warning("TensorFlow version does not support 'enable_op_determinism'.")

        # --- 2. Set execution mode (CPU/GPU) ---
        if execution_mode.lower() == "cpu":
            tf.config.set_visible_devices([], "GPU")
            logger.info("Training in CPU mode.")
            return tf.distribute.get_strategy()

        # --- 3. Select GPUs ---
        if isinstance(num_gpu, int):
            gpu_list = [num_gpu] if num_gpu >= 0 else []
        elif isinstance(num_gpu, (list, tuple)):
            gpu_list = list(num_gpu)
        else:
            raise ValueError("num_gpu must be int or list of ints.")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
        physical_gpus = tf.config.list_physical_devices("GPU")

        if physical_gpus:
            for gpu in physical_gpus:
                if vram_usage.lower() == "limit" and vram_limit_mb > 0:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=vram_limit_mb)]
                    )
                    logger.info(f"Limiting GPU memory to {vram_limit_mb}MB.")
                elif vram_usage.lower() == "growth":
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Enabled memory growth.")
                elif vram_usage.lower() == "full":
                    logger.info("Using full GPU memory.")
                else:
                    logger.warning(f"Unknown vram_usage mode: {vram_usage}. Using default.")

        # --- 4. Mixed Precision Training ---
        allowed_precisions = {
            "fp16": "mixed_float16",
            "float16": "mixed_float16",
            "bf16": "mixed_bfloat16",
            "bfloat16": "mixed_bfloat16",
            "mixed_float16": "mixed_float16",
            "mixed_bfloat16": "mixed_bfloat16",
        }
        # if mixed_precision_dtype:
        #     dtype = mixed_precision_dtype.lower()
        #     if dtype in allowed_precisions:
        #         from tensorflow.keras import mixed_precision
        #         policy = allowed_precisions[dtype]
        #         mixed_precision.set_global_policy(policy)
        #         logger.info(f"Using mixed precision: {policy}")
        #     else:
        #         logger.warning(f"Unsupported mixed_precision_dtype: {mixed_precision_dtype}")

        # --- 5. Execution Mode (Eager / Graph) ---
        if execution_mode.lower() == "eager":
            tf.config.run_functions_eagerly(True)
            logger.info("Execution mode: eager.")
        elif execution_mode.lower() == "graph":
            tf.config.run_functions_eagerly(False)
            logger.info("Execution mode: graph.")
        else:
            logger.error(f'Invalid execution_mode: {execution_mode}. Must be "eager", "graph", or "cpu".')
            return None

        if len(gpu_list) <= 1:
            strategy = tf.distribute.get_strategy()  # single-GPU or CPU
            logger.info("Using default strategy (1 GPU or CPU).")
        else:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using MirroredStrategy with {len(gpu_list)} GPUs.")

        return strategy

    except Exception as e:
        logger.error(f"[Train prepare error] {e}")
        return None


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
        