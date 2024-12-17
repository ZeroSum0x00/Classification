import tensorflow as tf


class GELULinear(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GELULinear, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        inputs_abs = tf.math.abs(inputs)
        inputs_sign = tf.math.sign(inputs)
        erf = inputs_abs * -0.7071
        erf = tf.nn.relu(erf + 1.769)
        erf = erf**2 * -0.1444 + 0.5
        return inputs * (erf * inputs_sign + 0.5)