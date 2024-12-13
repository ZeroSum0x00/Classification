import tensorflow as tf
from tensorflow.keras import metrics


class CategoricalAccuracy(tf.keras.metrics.Mean):
    def __init__(self, 
                 name='acc'):
        super().__init__(name=name)
        self.metric = metrics.CategoricalAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        super().update_state(self.metric(y_true, y_pred), sample_weight)


class TopKCategoricalAccuracy(tf.keras.metrics.Mean):
    def __init__(self, 
                 k=5,
                 name='acc'):
        name = name + f'_top_{k}'
        super().__init__(name=name)
        self.metric = metrics.TopKCategoricalAccuracy(k=k)
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        super().update_state(self.metric(y_true, y_pred), sample_weight)