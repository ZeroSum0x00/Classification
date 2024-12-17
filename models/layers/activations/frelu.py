import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from ..normalizers import get_normalizer_from_name


class FReLU(tf.keras.layers.Layer):
    """ FReLU activation https://arxiv.org/abs/2007.11824 """

    def __init__(self, kernel=(3, 3), **kwargs):
        super(FReLU, self).__init__(**kwargs)
        self.kernel = kernel
        
    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.conv = Conv2D(filters=out_dim, 
                           kernel_size=self.kernel, 
                           strides=(1, 1),
                           padding="same",
                           groups=out_dim,
                           use_bias=False)
        self.bn = get_normalizer_from_name('batch-norm')
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.bn(x, training= training)
        return tf.math.maximum(inputs, x)

    def get_config(self):
        config = super().get_config()
        config.update({
                "kernel": self.kernel,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
