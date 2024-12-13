import tensorflow as tf
from tensorflow.keras import metrics


class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, 
                 name='accuracy'):
        super().__init__(name=name)
        self.metric = metrics.BinaryAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(y_true)
        super().update_state(self.metric(y_true, y_pred), sample_weight)