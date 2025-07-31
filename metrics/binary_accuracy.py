import tensorflow as tf


class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.BinaryAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        depth = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(indices=y_true, depth=depth)
        y_pred = tf.cast(y_pred, tf.float32)
        self.acc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.acc.result()

    def reset_state(self):
        self.acc.reset_state()
        