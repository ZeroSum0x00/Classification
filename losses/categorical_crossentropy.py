import tensorflow as tf
from tensorflow.keras import losses


class CategoricalCrossentropy(losses.Loss):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy"
    ):
        super(CategoricalCrossentropy, self).__init__(name=name)
        self.losses = losses.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
        )
        self.invariant_name = "categorical_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        loss = self.losses(y_true, y_pred, sample_weight=sample_weight)
        return loss * self.coefficient
