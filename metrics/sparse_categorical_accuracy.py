import tensorflow as tf



class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        self.acc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.acc.result()

    def reset_state(self):
        self.acc.reset_state()



class SparseTopKCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k=5, name="accuracy", **kwargs):
        name = name + f"_top_{k}"
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)
        self.save_type = "increase"

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        self.acc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.acc.result()

    def reset_state(self):
        self.acc.reset_state()
        