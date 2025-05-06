import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils.time import datetime2string
from utils.logger import logger


class TrainSummary(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(TrainSummary, self).__init__()
        self.file_path = file_path
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        self.start_time = 0
        self.stop_time = 0
        self.delta_time = 0
        self.loss_infomation = {}
        self.metric_infomation = {}                 

    def on_epoch_end(self, epoch, logs={}):
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")

        if "loss" not in self.loss_infomation:
            self.loss_infomation["loss"] = {}
            self.loss_infomation["loss"]["train_values"] = []
            self.loss_infomation["loss"]["valid_values"] = []
        else:
            self.loss_infomation["loss"]["train_values"].append(train_loss)
            self.loss_infomation["loss"]["valid_values"].append(valid_loss)

        if self.model.list_metrics:
            for metric in self.model.list_metrics:
                metric_name = metric.name
                metric_type = metric.save_type.lower()
                if metric_name not in self.metric_infomation:
                    self.metric_infomation[metric_name] = {}
                    self.metric_infomation[metric_name]["train_values"] = []
                    self.metric_infomation[metric_name]["valid_values"] = []
    
                train_value = logs.get(metric_name)
                valid_value = logs.get("val_" + metric_name)
                self.metric_infomation[metric_name]["train_values"].append(train_value)
                self.metric_infomation[metric_name]["valid_values"].append(valid_value)
                self.metric_infomation[metric_name]["metric_type"]= metric_type
        
    def on_train_begin(self, epoch, logs={}):
        self.start_time = datetime.now()

    def on_train_end(self, epoch, logs={}):
        self.stop_time  = datetime.now()
        self.delta_time = self.stop_time - self.start_time

        write_data = "\tSummary training results\n\n"
        write_data += f"- Time train: {datetime2string(self.delta_time)}\n"

        for key, value in self.loss_infomation.items():
            loss_text = ""
            train_value = np.min(value["train_values"])
            loss_text += f"- Min train {key}: {train_value:.4f}"
            
            try:
                valid_value = np.min(value["valid_values"])
                loss_text += f", Min validation {key}: {valid_value:.4f}"
            except:
                pass
            write_data += loss_text + "\n"

        for key, value in self.metric_infomation.items():
            metric_text = ""
            if value["metric_type"] == "increase":
                train_value = np.max(value["train_values"])
                metric_text += f"- Max train {key}: {train_value:.4f}"
            elif value["metric_type"] == "decrease":
                train_value = np.min(value["train_values"])
                metric_text += f"- Min train {key}: {train_value:.4f}"
                
            try:
                if value["metric_type"] == "increase":
                    valid_value = np.max(value["valid_values"])
                    metric_text += f", Max validation {key}: {valid_value:.4f}"
                elif value["metric_type"] == "decrease":
                    valid_value = np.min(value["valid_values"])
                    metric_text += f", Min validation {key}: {valid_value:.4f}"
            except:
                pass
            write_data += metric_text + "\n"

        with open(self.file_path, "w") as f:
            f.write(write_data)
