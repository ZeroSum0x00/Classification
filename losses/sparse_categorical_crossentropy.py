import tensorflow as tf
from tensorflow.keras import losses


class SparseCategoricalCrossentropy(losses.Loss):
    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction="sum_over_batch_size",
        name="sparse_categorical_crossentropy"
    ):
        super(SparseCategoricalCrossentropy, self).__init__(name=name)
        self.losses = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            ignore_class=ignore_class,
            reduction=reduction,
        )
        self.invariant_name = "sparse_categorical_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.losses(y_true, y_pred, sample_weight=sample_weight)
        return loss * self.coefficient
