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


class AccuracyHistory(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path=None, 
                 save_best=False,
                 min_ratio=0.2):
        super(AccuracyHistory, self).__init__()

        self.result_path            = result_path
        self.save_best              = save_best
        self.min_ratio              = min_ratio
        
        self.train_accuracy_list    = []
        self.valid_accuracy_list    = []
        self.epoches                = [0]
        self.current_train_accuracy = 0.0
        self.current_valid_accuracy = 0.0
   
    def on_epoch_end(self, epoch, logs={}):
        train_accuracy = logs.get('categorical_accuracy')
        valid_accuracy = logs.get('val_categorical_accuracy')
        self.train_accuracy_list.append(train_accuracy)
        self.valid_accuracy_list.append(valid_accuracy)
        
        with open(os.path.join(self.result_path, "train_accuracy.txt"), 'a') as f:
            f.write(f"Train accuracy in epoch {epoch + 1}: {str(logs.get('categorical_accuracy'))}")
            f.write("\n")
        with open(os.path.join(self.result_path, "val_accuracy.txt"), 'a') as f:
            f.write(f"Train accuracy in epoch {epoch + 1}: {str(logs.get('val_categorical_accuracy'))}")
            f.write("\n")
            
        iters = range(len(self.train_accuracy_list))

        plt.figure()
        plt.plot(iters, self.train_accuracy_list, 'red', linewidth = 2, label='train accuracy')
        plt.plot(iters, self.valid_accuracy_list, 'coral', linewidth = 2, label='valid accuracy')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('A Accuracy Curve')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.result_path, "epoch_accuracy.png"))

        plt.cla()
        plt.close("all")
        
        if self.save_best:
            print('')
            if train_accuracy > self.current_train_accuracy and train_accuracy > self.min_ratio:
                logger.info(f'Train accuracy score increase {self.current_train_accuracy*100:.2f}% to {train_accuracy*100:.2f}%')
                logger.info(f'Save best train accuracy weights to {self.result_path}best_train_accuracy')
                self.model.save_weights(self.result_path + f'best_train_accuracy')
                self.current_train_accuracy = train_accuracy
            if valid_accuracy and valid_accuracy > self.current_valid_accuracy and valid_accuracy > self.min_ratio:
                logger.info(f'Validation accuracy score increase {self.current_valid_accuracy*100:.2f}% to {valid_accuracy*100:.2f}%')
                logger.info(f'Save best validation accuracy weights to {self.result_path}best_valid_accuracy')
                self.model.save_weights(self.result_path + f'best_valid_accuracy')
                self.current_valid_accuracy = valid_accuracy
            if train_accuracy == 1.0 and valid_accuracy and valid_accuracy > self.current_valid_accuracy:
                logger.info(f'Train accuracy is 100%, but validation accuracy increase to {valid_accuracy*100:.2f}%')
                logger.info(f'Save best train accuracy weights to {self.result_path}best_train_accuracy')
                self.model.save_weights(self.result_path + f'best_train_accuracy')
            if valid_accuracy == 1.0 and valid_accuracy and train_accuracy > self.current_train_accuracy:
                logger.info(f'Validation accuracy is 100%, but train accuracy increase to {train_accuracy*100:.2f}%')
                logger.info(f'Save best validation accuracy weights to {self.result_path}best_valid_accuracy')
                self.model.save_weights(self.result_path + f'best_valid_accuracy')
