import tensorflow as tf


class ReduceWrapper(tf.keras.layers.Layer):
    def __init__(self, reduce_mode='mean', axis=1, *args, **kwargs):
        super(ReduceWrapper, self).__init__(*args, **kwargs)
        self.reduce_mode = reduce_mode
        self.axis = axis

    def call(self, inputs):
        if self.reduce_mode.lower() == "all":
            return tf.reduce_all(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "any":
            return tf.reduce_any(inputs, axis=self.axis)
        elif self.reduce_mode.lower() in ["euclidean", "euclideannorm", "euclidean-norm"]:
            return tf.reduce_euclidean_norm(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "logsumexp":
            return tf.reduce_logsumexp(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "max":
            return tf.reduce_max(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "mean":
            return tf.reduce_mean(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "min":
            return tf.reduce_min(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "prod":
            return tf.reduce_prod(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "std":
            return tf.reduce_std(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "sum":
            return tf.reduce_sum(inputs, axis=self.axis)
        elif self.reduce_mode.lower() == "variance":
            return tf.reduce_variance(inputs, axis=self.axis)
        else:
            raise ValueError('reduce_mode mutch in ["all", "any", "euclidean", "logsumexp", "max", "mean", "min", "prod", "std", "sum", "variance]')
        