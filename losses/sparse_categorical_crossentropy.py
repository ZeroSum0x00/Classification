import tensorflow as tf
from tensorflow.keras import losses


class SparseCategoricalCrossentropy(losses.Loss):
    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction="sum_over_batch_size",
        name=None,
    ):
        super(SparseCategoricalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.losses = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            ignore_class=ignore_class,
        )
        self.invariant_name = "sparse_categorical_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.losses(y_true, y_pred)
        return loss * self.coefficient
