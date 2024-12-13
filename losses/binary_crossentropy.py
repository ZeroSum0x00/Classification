import tensorflow as tf
from tensorflow.keras import losses


class BinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, 
                 from_logits=False,
                 label_smoothing=0.0,
                 reduction='sum_over_batch_size', 
                 name=None):
        super(BinaryCrossentropy, self).__init__(reduction=reduction, name=name)
        self.losses = losses.BinaryCrossentropy(from_logits=from_logits,
                                                label_smoothing=label_smoothing)
        self.invariant_name = "binary_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.losses(y_true, y_pred)
        return loss * self.coefficient