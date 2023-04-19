import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.signal
from utils.logger import logger
from configs import general_config as cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path=None, 
                 saved_best_loss=False):
        super(LossHistory, self).__init__()

        self.result_path          = result_path
        self.saved_best_loss      = saved_best_loss
        
        self.train_losses         = []
        self.val_losses           = []
        self.epoches              = [0]
        self.current_val_loss     = 0.0
   
    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        with open(os.path.join(self.result_path, "train_loss.txt"), 'a') as f:
            f.write(f"Train loss in epoch {epoch + 1}: {str(logs.get('loss'))}")
            f.write("\n")
        with open(os.path.join(self.result_path, "val_loss.txt"), 'a') as f:
            f.write(f"Train loss in epoch {epoch + 1}: {str(logs.get('val_loss'))}")
            f.write("\n")
            
        iters = range(len(self.train_losses))

        plt.figure()
        plt.plot(iters, self.train_losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.train_losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.train_losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.result_path, "epoch_loss.png"))

        plt.cla()
        plt.close("all")