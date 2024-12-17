import tensorflow as tf


class MemoryEfficientMish(tf.keras.layers.Layer):
    """  Mish activation memory-efficient """

    def __init__(self, **kwargs):
        super(MemoryEfficientMish, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        if training:
            sx = tf.keras.backend.sigmoid(inputs)
            fx = tf.math.softplus(inputs)
            fx = tf.math.tanh(fx)
            return fx + inputs * sx * (1 - fx * fx)
        else:
            return inputs * tf.math.tanh(tf.math.softplus(inputs))