import os
import random
import datetime
import numpy as np
import tensorflow as tf
from .logger import logger



def train_prepare(
    execution_mode="graph",
    device=None,
    mixed_precision_dtype=None,
    init_seed=-1
):
    try:
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

        device_kine = device.get("kine", "gpu") if device else "cpu"
        num_usage = device.get("num_usage", -1) if device else -1
        vram_usage = device.get("vram_usage", "full") if device_kine == "gpu" else None
        vram_limit_mb = device.get("vram_limit_mb", 10240) if device_kine == "gpu" else None

        # --- 2. Set execution mode (CPU/GPU) ---
        if device_kine.lower() == "cpu":
            tf.config.set_visible_devices([], "GPU")

            if num_usage <= 0:
                num_threads = os.cpu_count()
            else:
                num_threads = num_usage

            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)

            logger.info(f"Training in CPU mode with {num_threads} threads.")
            return tf.distribute.get_strategy()

        # --- 3. Select GPUs ---
        if isinstance(num_usage, int):
            if num_usage == -1:
                gpu_list = None
            elif num_usage >= 0:
                gpu_list = [num_usage]
            else:
                raise ValueError("num_usage must be -1 or a non-negative int when using GPU.")
        elif isinstance(num_usage, (list, tuple)):
            gpu_list = list(num_usage)
        else:
            raise ValueError("num_usage must be int or list of ints.")
        
        if gpu_list is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
        physical_gpus = tf.config.list_physical_devices("GPU")
        if gpu_list is None:
            gpu_list = list(range(len(physical_gpus)))

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
        if mixed_precision_dtype:
            dtype = mixed_precision_dtype.lower()
            if dtype in allowed_precisions:
                from tensorflow.keras import mixed_precision
                policy = allowed_precisions[dtype]
                try:
                    mixed_precision.set_global_policy(policy)
                    logger.info(f"Using mixed precision: {policy}")
                except Exception as exc:
                    logger.error(f"Failed to set mixed precision policy {policy}: {exc}")
            else:
                logger.warning(f"Unsupported mixed_precision_dtype: {mixed_precision_dtype}")

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
            strategy = tf.distribute.get_strategy()
            logger.info("Using default strategy.")
        else:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using mirrored strategy with {len(gpu_list)} GPUs.")

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
    TRAINING_TIME_PATH = os.path.join(saved_dir, current_time)
    access_rights = 0o755
    try:
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        logger.info("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return TRAINING_TIME_PATH
    except: 
        logger.error("Creation of the directory %s failed" % TRAINING_TIME_PATH)
        