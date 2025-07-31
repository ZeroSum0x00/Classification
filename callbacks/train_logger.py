import os
import logging
import tensorflow as tf



class TrainLogger(tf.keras.callbacks.Callback):

    """Logs training metrics to a file.
    
    Args:
        file_path (str): Path to the log file will be saved.
        logLevel (int, optional): Logging level. Defaults to logging.INFO.
    """
    
    def __init__(self, file_path, logLevel=logging.INFO, console_output=False):
        super().__init__()
        self.file_path = file_path
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter("%(asctime)s - %(message)s")

        self.file_handler = logging.FileHandler(self.file_path)
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        if not console_output:
            self.logger.handlers[:] = []

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict=None):
        epoch_message = f'Epoch {epoch + 1}/{self.params["epochs"]}\n\t'
        logs_message = " - ".join([f"{key}: {value:.4f}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)
        