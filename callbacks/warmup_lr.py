import os
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class WarmUpLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 steps_per_epoch, 
                 epochs,
                 lr_init, 
                 lr_end,
                 warmup_epochs):
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.lr_init = lr_init
        self.lr_end = lr_end

    def on_train_batch_end(self, batch, logs=None):
        self.global_steps.assign_add(1)
        if self.global_steps < self.warmup_steps:
            lr = self.global_steps / self.warmup_steps * self.lr_init
        else:
            lr = self.lr_end + 0.5 * (self.lr_init - self.lr_end)*((1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi)))
        self.model.optimizer.lr.assign(lr.numpy())
        return self.global_steps.numpy(), self.model.optimizer.lr.numpy()


class AdvanceWarmUpLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path        = None,
                 lr_init            = 0.01,
                 lr_end             = 0.001,
                 epochs             = 100,
                 warmup_epoch_ratio = 0., 
                 warmup_lr_ratio    = 0.,
                 no_aug_epoch_ratio = 0.):
        self.result_path         = result_path
        self.result_path         = os.path.join(result_path, 'summary')
        self.lr_init             = lr_init
        self.lr_end              = lr_end
        self.epochs              = epochs
        self.warmup_total_epochs = min(max(warmup_epoch_ratio * epochs, 1), 3)
        self.warmup_lr_start     = max(warmup_lr_ratio * lr_init, 1e-6)
        self.no_aug_epoch        = min(max(no_aug_epoch_ratio * epochs, 1), 15)
        self.lr_list             = [0]
        self.epochs_list         = [0]
        os.makedirs(self.result_path, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.lr_init
        if epoch <= self.warmup_total_epochs:
            lr = (lr - self.warmup_lr_start) * pow(epoch / float(self.warmup_total_epochs), 2) + self.warmup_lr_start
        elif epoch >= self.epochs - self.no_aug_epoch:
            lr = self.lr_end
        else:
            lr = self.lr_end + 0.5 * (lr - self.lr_end) * (1.0 + tf.cos(np.pi * (epoch - self.warmup_total_epochs) / (self.epochs - self.warmup_total_epochs - self.no_aug_epoch)))
        self.model.optimizer.lr.assign(lr)
        
        if self.result_path:
            self.lr_list.append(lr)
            self.epochs_list.append(epoch + 1)

            with open(os.path.join(self.result_path, "learning_rate.txt"), 'a') as f:
                try:
                    f.write(f"Learning rate in epoch {epoch + 1}: {str(lr.numpy())}")
                except:
                    f.write(f"Learning rate in epoch {epoch + 1}: {str(lr)}")
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epochs_list, self.lr_list, 'red', linewidth=2, label='learning rate')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.result_path, "learning_rate.png"))
            plt.cla()
            plt.close("all")
        return lr


class BasicReduceLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 lr_init, 
                 lr_end, 
                 epochs, 
                 num_steps = 10):
        self.decay_rate  = (lr_end / lr_init) ** (1 / (num_steps - 1))
        step_size   = epochs / num_steps

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        n       = epoch // self.step_size
        out_lr  = lr * self.decay_rate ** n
        self.model.optimizer.lr.assign(lr.numpy())
        return lr.numpy()