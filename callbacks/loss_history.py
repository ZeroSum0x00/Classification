import os
import numpy as np
import scipy.signal
import tensorflow as tf
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from visualizer import value_above_line


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        result_path=None, 
        max_ratio=1.0,
        save_best=False,
        save_mode="weights",
        save_head=True,
        run_mode="epoch",
        show_frequency=1):
        super(LossHistory, self).__init__()
        self.result_path = result_path
        self.max_ratio = max_ratio
        self.run_mode = run_mode
        self.save_best = save_best
        self.save_mode = save_mode
        self.save_head = save_head
        self.show_frequency = show_frequency
        self.train_loss_list = []
        self.valid_loss_list = []
        self.current_train_loss = 0.0
        self.current_valid_loss = 0.0
        self.weight_path = os.path.join(self.result_path, 'weights')
        self.summary_path = os.path.join(self.result_path, 'summary')
        os.makedirs(self.weight_path, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)
        
        if self.save_mode not in ["model", "weights"]:
            raise ValueError(f'Invalid input: {self.save_mode}. Expected values are ["model", "weights"].')

    def call_calc_loss(self, iters, logs):
        train_loss = logs.get('loss')
        valid_loss = logs.get('val_loss')
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
            
        iters = range(len(self.train_loss_list))

        f = plt.figure()
        max_height = max(np.max(self.train_loss_list), np.max(self.valid_loss_list) if np.any(self.valid_loss_list) else 0)
        max_width  = np.max(iters)
        value_above_line(
            f=f,
            x=iters,
            y=self.train_loss_list,
            i=np.argmin(self.train_loss_list),
            max_size=[max_height, max_width],
            linewidth=2,
            line_color='red',
            label=f'train loss',
        )
        
        if np.any(self.valid_loss_list):
            value_above_line(
                f=f,
                x=iters,
                y=self.valid_loss_list,
                i=np.argmin(self.valid_loss_list),
                max_size=[max_height, max_width],
                linewidth=2,
                line_color='coral',
                label=f'valid loss',
            )
        
        try:
            if len(self.train_loss_list) < 25:
                num = 5
            else:
                num = 15
            window_length = min(len(self.train_loss_list), num)
            polyorder = min(3, window_length - 1)
            
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.train_loss_list, window_length, polyorder),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss',
            )
            
            if np.any(self.valid_loss_list):
                plt.plot(
                    iters,
                    scipy.signal.savgol_filter(self.valid_loss_list, window_length, polyorder),
                    '#8B4513',
                    linestyle='--',
                    linewidth=2,
                    label='smooth valid loss',
                )
                
        except:
            pass

        plt.grid(True)
        plt.xlabel(f'{self.run_mode.capitalize()} iters')
        plt.ylabel('Loss values')
        plt.title('A Loss Curve')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(loc="upper left")

        plt.savefig(os.path.join(self.summary_path, f"{self.run_mode.lower()}_loss.png"))

        plt.cla()
        plt.close("all")
        
        if self.save_best:
            print('')
            if train_loss < self.current_train_loss and train_loss < self.max_ratio:
                logger.info(f'Train loss score increase {self.current_train_loss:.2f}% to {train_loss:.2f}%')
                self.current_train_loss = train_loss
                if self.save_mode == "model":
                    weight_path = os.path.join(self.weight_path, f"best_train_los.keras")
                    logger.info(f'Save best train loss model to {weight_path}')
                    self.model.save_model(weight_path, save_head=self.save_head)
                elif self.save_mode == "weights":
                    weight_path = os.path.join(self.weight_path, f"best_train_los.weights.h5")
                    logger.info(f'Save best train loss weights to {weight_path}')
                    self.model.save_weights(weight_path, save_head=self.save_head)
                    
            if valid_loss < self.current_valid_loss and valid_loss < self.max_ratio:
                logger.info(f'Validation loss score increase {self.current_valid_loss:.2f}% to {valid_loss:.2f}%')
                self.current_valid_loss = valid_loss
                if self.save_mode == "model":
                    weight_path = os.path.join(self.weight_path, f"best_valid_loss.keras")
                    logger.info(f'Save best validation loss model to {weight_path}')
                    self.model.save_model(weight_path, save_head=self.save_head)
                elif self.save_mode == "weights":
                    weight_path = os.path.join(self.weight_path, f"best_valid_loss.weights.h5")
                    logger.info(f'Save best validation loss weights to {weight_path}')
                    self.model.save_weights(weight_path, save_head=self.save_head)
                
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