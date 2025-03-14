import tensorflow as tf
from tensorflow.keras import metrics


class SparseCategoricalAccuracy(tf.keras.metrics.Mean):
    def __init__(self,  
                 name='accuracy'):
        super().__init__(name=name)
        self.metric = metrics.SparseCategoricalAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(self.metric(y_true, y_pred), sample_weight)


class SparseTopKCategoricalAccuracy(tf.keras.metrics.Mean):
    def __init__(self,
                 k=5,
                 name='accuracy'):
        name = name + f'_top_{k}'
        super().__init__(name=name)
        self.metric = metrics.SparseTopKCategoricalAccuracy(k=k)
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(self.metric(y_true, y_pred), sample_weight)