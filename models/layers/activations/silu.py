import tensorflow as tf


class SiLU(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SiLU, self).__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=False):
        return inputs * tf.keras.backend.sigmoid(inputs)
    