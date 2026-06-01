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
from utils.post_processing import get_labels
from utils.config_processing import load_config
from utils.logger import logger



def train(engine_file_config, model_file_config):
    engine_config = load_config(engine_file_config)
    model_config = load_config(model_file_config)
    data_config = engine_config["Dataset"]
    model_config = model_config["Model"]
    train_config = engine_config["Train"]
    loss_config = engine_config["Losses"]
    optimizer_config = engine_config["Optimizer"]
    metric_config = engine_config["Metrics"]
    callbacks_config = engine_config["Callbacks"]

    strategy = train_prepare(
        execution_mode=train_config["strategy"].get("execution_mode", "graph"),
        device=train_config.get("device", None),
        mixed_precision_dtype=train_config["advanced"].get("mixed_precision_dtype"),
        init_seed=train_config["advanced"].get("random_seed", 42),
    )

    if strategy is None:
        raise RuntimeError("Failed to prepare training environment.")
        
    train_strategy = train_config["strategy"].get("train_mode", "scratch")
    start_epoch = train_config["hyperparams"]["epoch"].get("start", 0)
    stop_epoch= train_config["hyperparams"]["epoch"].get("end", 100)

    TRAINING_TIME_PATH = create_folder_weights(train_config.get("output_path", "saved_weights"))
    shutil.copy(model_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(model_file_config)))
    shutil.copy(engine_file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(engine_file_config)))

    if not model_config["classes"]:
        classes, _ = get_labels(data_config["overall"]["source_paths"])
        model_config["classes"] = classes
        model_config["is_set_classes"] = True
    
    model_config["train_strategy"] = train_config.get("train_strategy", "scratch")

    batch_size = train_config["hyperparams"]["batch_size"] if train_config["hyperparams"]["batch_size"] != -1 else find_max_batch_size(model)
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    optimizer_config["learning_rate"] = optimizer_config["learning_rate"] / strategy.num_replicas_in_sync

    with open(os.path.join(TRAINING_TIME_PATH, "classes.names"), "w") as f:
        for cls in classes:
            f.write(cls + "\n")

    data_generator_instance = get_train_test_data(
        dataloader_mode=data_config.get("dataloader_mode", "tf"),
        load_subset=data_config["overall"].get("load_subset", ["train", "valid"]),
        data_source_paths=data_config["overall"]["source_paths"],
        classes=classes,
        target_size=model_config["inputs"],
        batch_size=global_batch_size,
        color_space=data_config["overall"].get("color_space", "RGB"),
        augmentor=data_config.get("augmentation", {}),
        normalizer=data_config.get("normalizer", {}),
        sampler=data_config["overall"].get("sampler"),
        data_type=data_config["overall"]["data_type"],
        check_data=data_config["overall"].get("check_data", False),
        load_memory=data_config["overall"].get("load_memory", False),
        num_workers=train_config["advanced"].get("num_workers", 1),
    )

    train_generator = data_generator_instance["train_generator"]
    valid_generator = data_generator_instance.get("valid_generator", None)
    test_generator = data_generator_instance.get("test_generator", None)
    
    train_step = int(np.ceil(train_generator.N / global_batch_size))
    train_generator = train_generator.get_dataset() if isinstance(train_generator, TFDataPipeline) else train_generator

    if valid_generator:
        valid_step = int(np.ceil(valid_generator.N / global_batch_size))
        valid_generator = valid_generator.get_dataset() if isinstance(valid_generator, TFDataPipeline) else valid_generator

    if test_generator:
        test_step = int(np.ceil(test_generator.N / global_batch_size))
        test_generator = test_generator.get_dataset() if isinstance(test_generator, TFDataPipeline) else test_generator
    
    with strategy.scope():
        losses = build_losses(loss_config)
        optimizer = build_optimizer(optimizer_config)
        metrics = build_metrics(metric_config)

        model = build_models(train_config, model_config)
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
            start_epoch = int(model.current_epoch)

    with strategy.scope():
        if valid_generator:
            model.fit(
                train_generator,
                steps_per_epoch=train_step,
                validation_data=valid_generator,
                validation_steps=valid_step,
                epochs=stop_epoch,
                initial_epoch=start_epoch,
                callbacks=callbacks,
                )
        else:
            model.fit(
                train_generator,
                steps_per_epoch=train_step,
                epochs=stop_epoch,
                initial_epoch=start_epoch,
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
        "--engine_config", type=str, default="./configs/engine/engine.yaml",
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
