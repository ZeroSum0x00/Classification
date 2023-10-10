import tensorflow as tf


class ReLU6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)
        self.activation = tf.keras.layers.ReLU(max_value=6.)
        
    def call(self, inputs, training=False):
        return self.activation(inputs)

        
class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))