import tensorflow as tf
from tensorflow.keras import losses



class BinaryCrossentropy(losses.Loss):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        crossentropy_axis=-1,
        reduction="mean",
        name="binary_crossentropy"
    ):
        super(BinaryCrossentropy, self).__init__(name=name)
        self.loss_fn = losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=crossentropy_axis,
            reduction=None,
        )
        self.reduction = reduction
        self.invariant_name = "binary_crossentropy"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        per_sample_loss = self.loss_fn(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, per_sample_loss.dtype)
            per_sample_loss *= sample_weight

        if self.reduction is None:
            loss_value = per_sample_loss
        elif self.reduction.lower() == "mean":
            loss_value = tf.reduce_mean(per_sample_loss)
        elif self.reduction.lower() == "sum":
            loss_value = tf.reduce_sum(per_sample_loss)
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

        return loss_value * self.coefficient
        