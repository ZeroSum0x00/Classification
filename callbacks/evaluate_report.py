import os
import io
import cv2
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.auxiliary_processing import fig_to_cv2_image
from utils.eval_caculator import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from visualizer import value_above_line, VideoRender
from utils.logger import logger


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(
        self,
        result_path=None,
        sample_weight=None,
        normalize_confusion_matrix=False,
        min_ratio=0.2,
        save_best=True,
        save_mode="weights",
        save_head=True,
        show_frequency=10,
    ):
        super(Evaluate, self).__init__()
        self.result_path = result_path
        self.sample_weight = sample_weight
        self.normalize_confusion_matrix = normalize_confusion_matrix
        self.min_ratio = min_ratio
        self.save_best = save_best
        self.save_mode = save_mode
        self.save_head = save_head
        self.show_frequency = show_frequency
        self.epoches = [0]
        self.metric_values = [0]
        self.current_value = 0.0
        self.eval_dataset = None
        self.video_render = VideoRender(frame_duration=0.7, save_path=os.path.join(result_path, 'summary', 'confusion_matrix_per_epoch.mp4'))

        if self.save_mode not in ["model", "weights"]:
            raise ValueError(f"Invalid input: {self.save_mode}. Expected values are ['model', 'weights'].")

    def pass_data(self, data):
        self.eval_dataset = data

    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        save_weight_path = os.path.join(self.result_path, "weights")
        os.makedirs(save_weight_path, exist_ok=True)
        summary_path = os.path.join(self.result_path, "summary")
        os.makedirs(summary_path, exist_ok=True)
        visualizer_path = os.path.join(self.result_path, "visualizer")
        os.makedirs(visualizer_path, exist_ok=True)
        
        if not hasattr(self, "classes"):
            self.classes = self.model.classes

        if temp_epoch % self.show_frequency == 0:
            if self.eval_dataset is not None and self.classes:
                logger.info("\nGet evaluate metrics report.")

                predictions = []
                gts = []
                for i, data in enumerate(tqdm(self.eval_dataset)):
                    images, labels, _ = data
                    preds = self.model.predict(images)
                    preds = np.argmax(preds, axis=1)
                    predictions.append(preds)
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
                            weight_path = os.path.join(save_weight_path, "best_eval_acc.keras")
                            logger.info(f"Save best evaluate accuracy model to {weight_path}")
                            self.model.save_model(weight_path, save_head=self.save_head)
                        elif self.save_mode == "weights":
                            weight_path = os.path.join(save_weight_path, "best_eval_acc.weights.h5")
                            logger.info(f"Save best evaluate accuracy weights to {weight_path}")
                            self.model.save_weights(weight_path, save_head=self.save_head)
                        
                self.epoches.append(temp_epoch)
                with open(os.path.join(summary_path, "evaluate_report.txt"), "a") as f:
                    f.write(f"Epoch {epoch + 1}: {accuracy}\n")

                
                # ===== Confusion Matrix =====
                epoch_visualizer_path = os.path.join(visualizer_path, f"epoch-{temp_epoch}")
                os.makedirs(epoch_visualizer_path, exist_ok=True)
                conf_mat_path = os.path.join(epoch_visualizer_path, "confusion_matrix.png")

                cm = confusion_matrix(gts, predictions, labels=range(len(self.classes)))
                if self.normalize_confusion_matrix:
                    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                    cm = np.nan_to_num(cm)
                    
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
                disp.plot(
                    cmap=plt.cm.Blues,
                    xticks_rotation=45,
                    values_format=".2f" if self.normalize_confusion_matrix else "d"
                )
                
                plt.title(f"{'Normalized ' if self.normalize_confusion_matrix else ''}Confusion Matrix - Epoch {temp_epoch}")
                plt.tight_layout()

                cm_array = fig_to_cv2_image(plt.gcf())
                self.video_render(cm_array)
                plt.savefig(conf_mat_path)
                plt.clf()
                plt.close()
                
                # ===== Accuracy Graph =====
                f = plt.figure()
                max_height = np.max(self.metric_values)
                max_width = np.max(self.epoches)
                
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
    
                plt.savefig(os.path.join(summary_path, "epoch_eval_acc.png"))
                plt.clf()
                plt.close("all")
                
    def on_train_end(self, logs=None):
        self.video_render.release()