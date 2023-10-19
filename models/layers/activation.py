import tensorflow as tf
from tensorflow.keras.layers import Activation


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


def get_activation_from_name(activ_name, *args, **kwargs):
    activ_name = activ_name.lower()
    if activ_name in ['relu', 'sigmoid', 'softmax', 'softplus', 'phish', 'hard_swish', 'gelu', 'swish']:
        return Activation(activ_name)
    elif activ_name == 'relu6':
        return ReLU6(*args, **kwargs)
    elif activ_name in ['leaky', 'leakyrelu', 'leaky-relu']:
        return LeakyReLU(*args, **kwargs)
    elif activ_name == 'mish':
        return Mish(*args, **kwargs)
    else:
        return Activation('linear')