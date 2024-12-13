import os
import scipy.signal
import tensorflow as tf

from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path    = None, 
                 max_ratio      = 1.0,
                 save_best      = False,
                 save_format    = 'tf',
                 run_mode       = "epoch",
                 show_frequency = 1):
        super(LossHistory, self).__init__()

        self.result_path        = result_path
        self.max_ratio          = max_ratio
        self.run_mode           = run_mode
        self.save_best          = save_best
        self.save_format        = save_format
        self.show_frequency     = show_frequency
        
        self.train_loss_list    = []
        self.valid_loss_list    = []
        self.current_train_loss = 0.0
        self.current_valid_loss = 0.0
        self.weight_path        = os.path.join(self.result_path, 'weights')
        os.makedirs(self.weight_path, exist_ok=True)
        self.summary_path       = os.path.join(self.result_path, 'summary')
        os.makedirs(self.summary_path, exist_ok=True)
        
    def call_calc_loss(self, iters, logs):
        train_loss = logs.get('loss')
        valid_loss = logs.get('val_loss')
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
            
        iters = range(len(self.train_loss_list))

        plt.figure()
        plt.plot(iters, self.train_loss_list, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.valid_loss_list, 'coral', linewidth = 2, label='valid loss')
        try:
            if len(self.train_loss_list) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.train_loss_list, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.valid_loss_list, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth valid loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel(f'{self.run_mode.capitalize()} iters')
        plt.ylabel('Loss values')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.summary_path, f"{self.run_mode.lower()}_loss.png"))

        plt.cla()
        plt.close("all")
        
        if self.save_best:
            print('')
            if train_loss < self.current_train_loss and train_loss < self.max_ratio:
                logger.info(f'Train loss score increase {self.current_train_loss:.2f}% to {train_loss:.2f}%')
                logger.info(f'Save best train loss weights to {os.path.join(self.weight_path, f"best_train_loss")}')
                self.model.save_weights(os.path.join(self.weight_path, f'best_train_loss'), save_format=self.save_format)
                self.current_train_loss = train_loss
            if valid_loss < self.current_valid_loss and valid_loss < self.max_ratio:
                logger.info(f'Validation loss score increase {self.current_valid_loss:.2f}% to {valid_loss:.2f}%')
                logger.info(f'Save best validation loss weights to {os.path.join(self.weight_path, f"best_valid_loss")}')
                self.model.save_weights(os.path.join(self.weight_path, f'best_valid_loss'), save_format=self.save_format)
                self.current_valid_loss = valid_loss
                
    def on_epoch_end(self, epoch, logs=None):
        accept_type = ['epoch']
        if self.run_mode.lower() in accept_type and epoch % self.show_frequency == 0:
            self.call_calc_loss(epoch, logs)
            
    def on_batch_end(self, batch, logs=None):
        accept_type = ['batch', 'step']
        if not hasattr(self, 'step_iter'):
            self.step_iter = 0
            
        if self.run_mode.lower() in accept_type and batch != 0 and batch % self.show_frequency == 0:
            self.call_calc_loss(self.step_iter, logs)
            self.step_iter += 1