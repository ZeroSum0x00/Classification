import tensorflow as tf
from tensorflow.keras import metrics


class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, 
                 name='accuracy'):
        super().__init__(name=name)
        self.metric = metrics.BinaryAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        super().update_state(self.metric(y_true, y_pred), sample_weight)