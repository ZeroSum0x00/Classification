import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend as K

class CategoricalCrossentropy(losses.Loss):
    def __init__(self, 
                 from_logits=False,
                 label_smoothing=0.0,
                 reduction='sum_over_batch_size', 
                 name=None):
        super(CategoricalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.losses = losses.CategoricalCrossentropy(from_logits=from_logits,
                                                     label_smoothing=label_smoothing)
        self.invariant_name = "categorical_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        loss = self.losses(y_true, y_pred)
        return loss * self.coefficient
