import tensorflow as tf
from .hardtanh import HardTanh


class HardSwish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.activation = HardTanh(min_val=0.0, max_val=6.0)

    def call(self, inputs, training=False):
        return inputs * self.activation(inputs + 3.0) / 6.0