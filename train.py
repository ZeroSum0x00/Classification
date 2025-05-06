import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import shutil
import argparse
import numpy as np
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from metrics import build_metrics
from callbacks import build_callbacks, Evaluate, CheckpointSaver
from data_utils import get_train_test_data, TFDataPipeline
from utils.train_processing import create_folder_weights, find_max_batch_size, train_prepare
from utils.config_processing import load_config
from utils.logger import logger


def train(engine_file_config, model_file_config):
    engine_config = load_config(engine_file_config)
    model_config  = load_config(model_file_config)
    data_config  = engine_config["Dataset"]
    model_config = model_config["Model"]
    train_config = engine_config["Train"]
    loss_config = engine_config["Losses"]
    optimizer_config = engine_config["Optimizer"]
    metric_config = engine_config["Metrics"]
    callbacks_config = engine_config["Callbacks"]

    if train_prepare(
        execution_mode=train_config.get("execution_mode", "graph"),
        vram_usage=train_config.get("vram_usage", "limit"),
        vram_limit_mb=train_config.get("vram_limit_mb", 10240),
        mixed_precision_dtype=train_config.get("mixed_precision_dtype"),
        num_gpu=train_config.get("num_gpus", 0),
        init_seed=train_config.get("random_seed", 42),
    ):
        train_strategy = train_config.get("train_strategy", "scratch")
        initial_epoch = train_config["epoch"].get("start", 0)
        TRAINING_TIME_PATH = create_folder_weights(train_config.get("output_path", "saved_weights"))
        shutil.copy(model_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(model_file_config)))
        shutil.copy(engine_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(engine_file_config)))
        
        if not model_config["classes"]:
            model_config["classes"] = data_config["data_dir"]

        model_config["train_strategy"] = train_config.get("train_strategy", "scratch")
        model = build_models(model_config)

        with open(os.path.join(TRAINING_TIME_PATH, "classes.names"), "w") as f:
            for cls in model.classes:
                f.write(cls + "\n")

        batch_size = find_max_batch_size(model) if train_config["batch_size"] == -1 else train_config["batch_size"]
        train_generator, valid_generator, test_generator = get_train_test_data(
            data_dirs=data_config["data_dir"],
            classes=model.classes,
            target_size=model_config["inputs"],
            batch_size=batch_size,
            color_space=data_config["data_info"].get("color_space", "RGB"),
            augmentor=data_config["data_augmentation"],
            normalizer=data_config["data_normalizer"].get("norm_type", "divide"),
            mean_norm=data_config["data_normalizer"].get("norm_mean"),
            std_norm=data_config["data_normalizer"].get("norm_std"),
            interpolation=data_config["data_normalizer"].get("interpolation", "BILINEAR"),
            data_type=data_config["data_info"]["data_type"],
            check_data=data_config["data_info"].get("check_data", False),
            load_memory=data_config["data_info"].get("load_memory", False),
            dataloader_mode=data_config.get("dataloader_mode", "tf"),
            get_data_mode=data_config.get("get_data_mode", 2),
            num_workers=train_config.get("num_workers", 1),
        )
        
        train_step = int(np.ceil(train_generator.N / batch_size))
        train_generator = train_generator.get_dataset() if isinstance(train_generator, TFDataPipeline) else train_generator

        if valid_generator:
            valid_step = int(np.ceil(valid_generator.N / batch_size))
            valid_generator = valid_generator.get_dataset() if isinstance(valid_generator, TFDataPipeline) else valid_generator

        if test_generator:
            test_step = int(np.ceil(test_generator.N / batch_size))
            test_generator  = test_generator.get_dataset() if isinstance(test_generator, TFDataPipeline) else test_generator

        losses    = build_losses(loss_config)
        optimizer = build_optimizer(optimizer_config)
        metrics   = build_metrics(metric_config)
        
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        
        callbacks = build_callbacks(callbacks_config, TRAINING_TIME_PATH)

        for callback in callbacks:
            if isinstance(callback, Evaluate):
                if test_generator:
                    callback.pass_data(test_generator)
                else:
                    callback.pass_data(valid_generator)

        if train_strategy not in ["feature-extraction", "fine-tuning"]:
            checkpoint_path = train_config.get("checkpoints") if train_config.get("checkpoints") and os.path.isdir(train_config.get("checkpoints")) else os.path.join(TRAINING_TIME_PATH, "checkpoints")
            ckpt = tf.train.Checkpoint(
                epoch=model.current_epoch,
                optimizer=model.optimizer,
                model=model
            )
    
            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
            callbacks = [CheckpointSaver(model, ckpt_manager)] + callbacks

            if ckpt_manager.latest_checkpoint:
                ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
                logger.info(f"Restored from {ckpt_manager.latest_checkpoint}")
                initial_epoch = int(model.current_epoch)
            
        if valid_generator:
            model.fit(
                train_generator,
                steps_per_epoch=train_step,
                validation_data=valid_generator,
                validation_steps=valid_step,
                epochs=train_config["epoch"]["end"],
                initial_epoch=initial_epoch,
                callbacks=callbacks,
            )
        else:
            model.fit(
                train_generator,
                steps_per_epoch=train_step,
                epochs=train_config["epoch"]["end"],
                initial_epoch=initial_epoch,
                callbacks=callbacks,
            )

        if test_generator:
            model.evaluate(
                test_generator,
                steps=test_step,
            )

        save_mode = train_config.get("model_save_mode", "weights")
        save_head = train_config.get("model_save_head", True)
        if save_mode.lower() == "model":
            weight_path = os.path.join(TRAINING_TIME_PATH, "weights", "last_weights.keras")
            logger.info(f"Save last model to {weight_path}")
            model.save_model(weight_path, save_head=save_head)
        elif save_mode.lower() == "weights":
            weight_path = os.path.join(TRAINING_TIME_PATH, "weights", "last_weights.weights.h5")
            logger.info(f"Save last weights to {weight_path}")
            model.save_weights(weight_path, save_head=save_head)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified config files.")
    parser.add_argument(
        "--engine_config", type=str, default="./configs/test/engine.yaml",
        help="Path to the engine configuration YAML file. Default: ./configs/test/engine.yaml"
    )
    parser.add_argument(
        "--model_config", type=str, default="./configs/test/model.yaml",
        help="Path to the model configuration YAML file. Default: ./configs/test/model.yaml"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args.engine_config, args.model_config)
