import os
import cv2
import shutil
import colorsys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.auxiliary_processing import change_color_space
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.post_processing import detect_images
from sklearn.metrics import classification_report
from visualizer import value_above_line

    
class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path     = None, 
                 min_ratio       = 0.2,
                 saved_best_map  = True,
                 show_frequency  = 100):
        super(Evaluate, self).__init__()
        self.result_path          = result_path
        self.min_ratio            = min_ratio
        self.saved_best_map       = saved_best_map
        self.show_frequency       = show_frequency
        self.epoches              = [0]
        self.metric_values        = [0]
        self.current_acc          = 0.0
        self.eval_dataset          = None


    def pass_data(self, data):
        self.eval_dataset = data

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self, "classes"):
            self.classes = self.model.classes

        temp_epoch = epoch + 1
        if temp_epoch % self.show_frequency == 0:
            if self.eval_dataset is not None and self.classes:
                print("\nGet report.")

                predictions = []
                gts         = []
                gts2        = []
                total_count = self.eval_dataset.N
                true_count  = 0
                for i, (images, labels) in enumerate(tqdm(self.eval_dataset)):
                    pred = self.model.predict(images)
                    pred = np.argmax(pred, axis=1)
                    predictions.append(pred)
                    gts.append(labels) 
                    
                    top1_list, result_list = detect_images(images, self.model.architecture, self.classes)
                    gts2.append([top1[0] for top1 in top1_list])

                predictions = tuple(item for sublist in predictions for item in sublist)
                gts = tuple(item for sublist in gts for item in sublist)
                gts2 = tuple(item for sublist in gts2 for item in sublist)
                print(classification_report(gts, predictions))

                for a, b in zip(gts2, gts):
                    if a == self.classes[b]:
                        true_count += 1

                mean_acc = true_count / total_count
                self.metric_values.append(mean_acc)
                print(f'Accuracy = {mean_acc * 100:.2f}%')

                if self.saved_best_map:
                    if mean_acc > self.current_acc and mean_acc > self.min_ratio:
                        logger.info(f'Evaluate accuracy score increase {self.current_acc*100:.2f}% to {mean_acc*100:.2f}%')
                        weight_path = os.path.join(self.result_path, "weights", "best_eval_acc.weights.h5")
                        logger.info(f'Save best evaluate accuracy weights to {weight_path}')                    
                        self.model.save_weights(weight_path)
                        self.current_acc = mean_acc
                        
                self.epoches.append(temp_epoch)
                with open(os.path.join(self.result_path, 'summary', "best_eval_acc.txt"), 'a') as f:
                    f.write(f"Epoch {epoch + 1}: {mean_acc}\n")

                f = plt.figure()
                max_height = np.max(self.metric_values)
                max_width  = np.max(self.epoches)
                value_above_line(f,
                                 x=self.epoches,
                                 y=self.metric_values,
                                 i=np.argmax(self.metric_values),
                                 max_size=[max_height, max_width],
                                 linewidth=2,
                                 line_color='red',
                                 label='acc mean')
                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Evaluate accuracy')
                plt.title('A evaluate accuracy graph')
                
                handles, labels = plt.gca().get_legend_handles_labels()
                if labels:
                    plt.legend(fontsize=7, loc="upper left")
    
                plt.savefig(os.path.join(self.result_path, 'summary', "epoch_eval_acc.png"))
                plt.cla()
                plt.close("all")