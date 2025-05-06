import tensorflow as tf
from tensorflow.keras import losses


class BinaryFocalCrossentropy(losses.Loss):
    def __init__(
        self,
        apply_class_balancing=False,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_focal_crossentropy"
    ):
        super(BinaryFocalCrossentropy, self).__init__(name=name)
        self.losses = losses.BinaryFocalCrossentropy(
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
        )
        self.invariant_name = "binary_focal_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.losses(y_true, y_pred, sample_weight=sample_weight)
        return loss * self.coefficient
