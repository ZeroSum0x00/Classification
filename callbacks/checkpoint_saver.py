import tensorflow as tf
from tensorflow.keras import callbacks
from utils.logger import logger



class CheckpointSaver(callbacks.Callback):
    def __init__(self, model, ckpt_manager):
        super().__init__()
        self.model_obj = model
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        self.model_obj.current_epoch.assign(epoch + 1)
        save_path = self.ckpt_manager.save()
        print("")
        logger.info(f"Saved checkpoint at: {save_path}")
