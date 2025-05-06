import tensorflow as tf


class Mish(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Mish, self).__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))
    