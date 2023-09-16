import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.signal
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class MetricHistory(tf.keras.callbacks.Callback):
    def __init__(self,
                 result_path=None, 
                 save_best=False,
                 min_ratio=0.2):
        super(MetricHistory, self).__init__()
        self.result_path            = result_path
        self.save_best              = save_best
        self.min_ratio              = min_ratio
        self.metric_infomation = {}

    def on_epoch_end(self, epoch, logs={}):
        for metric in self.model.list_metrics:
            metric_name = metric.name
            if metric_name not in self.metric_infomation:
                self.metric_infomation[metric_name] = {}
                self.metric_infomation[metric_name]['train_object'] = []
                self.metric_infomation[metric_name]['valid_object'] = []
                self.metric_infomation[metric_name]['train_value'] = 0.0
                self.metric_infomation[metric_name]['valid_value'] = 0.0

            train_value = logs.get(metric_name)
            valid_value = logs.get('val_' + metric_name)
            self.metric_infomation[metric_name]['train_object'].append(train_value)
            self.metric_infomation[metric_name]['valid_object'].append(valid_value)
            
            with open(os.path.join(self.result_path, f"train_{metric_name}.txt"), 'a') as f:
                f.write(f"Train {metric_name} in epoch {epoch + 1}: {str(train_value)}")
                f.write("\n")
            with open(os.path.join(self.result_path, f"val_{metric_name}.txt"), 'a') as f:
                f.write(f"Valid {metric_name} in epoch {epoch + 1}: {str(valid_value)}")
                f.write("\n")
                
            iters = range(len(self.metric_infomation[metric_name]['train_object']))
    
            plt.figure()
            plt.plot(iters, self.metric_infomation[metric_name]['train_object'], 'red', linewidth=2, label=f'train {metric_name.replace("-", " ")}')
            plt.plot(iters, self.metric_infomation[metric_name]['valid_object'], 'coral', linewidth=2, label=f'valid {metric_name.replace("-", " ")}')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.replace("-", " ").title())
            plt.title(f'A {metric_name.replace("-", " ").title()} Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.result_path, f"epoch_{metric_name}.png"))
            plt.cla()
            plt.close("all")
            
            if self.save_best:
                print('')
                if train_value > self.metric_infomation[metric_name]['train_value'] and train_value > self.min_ratio:
                    logger.info(f"Train {metric_name} score increase {self.metric_infomation[metric_name]['train_value']:.2f} to {train_value:.2f}")
                    logger.info(f'Save best train {metric_name} weights to {self.result_path}best_train_{metric_name}')
                    self.model.save_weights(self.result_path + f'best_train_{metric_name}')
                    self.metric_infomation[metric_name]['train_value'] = train_value
                if valid_value and valid_value > self.metric_infomation[metric_name]['valid_value'] and valid_value > self.min_ratio:
                    logger.info(f"Validation {metric_name} score increase {self.metric_infomation[metric_name]['valid_value']:.2f} to {valid_value:.2f}")
                    logger.info(f'Save best validation {metric_name} weights to {self.result_path}best_valid_{metric_name}')
                    self.model.save_weights(self.result_path + f'best_valid_{metric_name}')
                    self.metric_infomation[metric_name]['valid_value'] = valid_value
                if train_value == 1.0 and valid_value and valid_value > self.metric_infomation[metric_name]['valid_value']:
                    logger.info(f'Train {metric_name} is maximum, but validation {metric_name} increase to {valid_value:.2f}')
                    logger.info(f'Save best train {metric_name} weights to {self.result_path}best_train_{metric_name}')
                    self.model.save_weights(self.result_path + f'best_train_{metric_name}')
                if valid_value == 1.0 and valid_value and train_value > self.metric_infomation[metric_name]['train_value']:
                    logger.info(f'Validation {metric_name} is maximum, but train {metric_name} increase to {train_value:.2f}')
                    logger.info(f'Save best validation {metric_name} weights to {self.result_path}best_valid_{metric_name}')
                    self.model.save_weights(self.result_path + f'best_valid_{metric_name}')
