import os
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.eval_caculator import classification_report
from visualizer import value_above_line
from utils.logger import logger


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(
        self,
        result_path=None,
        sample_weight=None,
        min_ratio=0.2,
        save_best=True,
        save_mode="weights",
        save_head=True,
        show_frequency=100,
    ):
        super(Evaluate, self).__init__()
        self.result_path = result_path
        self.sample_weight = sample_weight
        self.min_ratio = min_ratio
        self.save_best = save_best
        self.save_mode = save_mode
        self.save_head = save_head
        self.show_frequency = show_frequency
        self.epoches = [0]
        self.metric_values = [0]
        self.current_value = 0.0
        self.eval_dataset = None
        
        if self.save_mode not in ["model", "weights"]:
            raise ValueError(f"Invalid input: {self.save_mode}. Expected values are ['model', 'weights'].")

    def pass_data(self, data):
        self.eval_dataset = data

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self, "classes"):
            self.classes = self.model.classes

        temp_epoch = epoch + 1
        if temp_epoch % self.show_frequency == 0:
            if self.eval_dataset is not None and self.classes:
                logger.info("\nGet evaluate metrics report.")

                predictions = []
                gts         = []
                for i, (images, labels) in enumerate(tqdm(self.eval_dataset)):
                    pred = self.model.predict(images)
                    pred = np.argmax(pred, axis=1)
                    predictions.append(pred)
                    gts.append(labels) 

                predictions = tuple(item for sublist in predictions for item in sublist)
                gts = tuple(item for sublist in gts for item in sublist)
                
                report_dict = classification_report(
                    y_true=gts,
                    y_pred=predictions,
                    target_names=self.classes,
                    sample_weight=self.sample_weight,
                    print_report=True,
                )
                
                class_report, sum_report = report_dict
                accuracy = sum_report.pop("accuracy", 0)
                self.metric_values.append(accuracy)

                if self.save_best:
                    if accuracy > self.current_value and accuracy > self.min_ratio:
                        logger.info(f"Evaluate accuracy score increase {self.current_value*100:.2f}% to {accuracy*100:.2f}%")
                        self.current_value = accuracy
                        if self.save_mode == "model":
                            weight_path = os.path.join(self.result_path, "weights", "best_eval_acc.keras")
                            logger.info(f"Save best evaluate accuracy model to {weight_path}")
                            self.model.save_model(weight_path, save_head=self.save_head)
                        elif self.save_mode == "weights":
                            weight_path = os.path.join(self.result_path, "weights", "best_eval_acc.weights.h5")
                            logger.info(f"Save best evaluate accuracy weights to {weight_path}")
                            self.model.save_weights(weight_path, save_head=self.save_head)
                        
                self.epoches.append(temp_epoch)
                with open(os.path.join(self.result_path, "summary", "evaluate_report.txt"), "a") as f:
                    f.write(f"Epoch {epoch + 1}: {accuracy}\n")

                f = plt.figure()
                max_height = np.max(self.metric_values)
                max_width  = np.max(self.epoches)
                
                value_above_line(
                    f=f,
                    x=self.epoches,
                    y=self.metric_values,
                    i=np.argmax(self.metric_values),
                    max_size=[max_height, max_width],
                    linewidth=2,
                    line_color="red",
                    label="acc mean",
                )
                
                plt.grid(True)
                plt.xlabel("Epoch")
                plt.ylabel("Evaluate accuracy")
                plt.title("A evaluate accuracy graph")
                
                handles, labels = plt.gca().get_legend_handles_labels()
                if labels:
                    plt.legend(fontsize=7, loc="upper left")
    
                plt.savefig(os.path.join(self.result_path, "summary", "epoch_eval_acc.png"))
                plt.cla()
                plt.close("all")
                