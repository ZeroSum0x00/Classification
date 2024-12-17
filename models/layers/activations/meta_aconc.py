import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class MetaAconC(tf.keras.layers.Layer):
    r""" ACON activation (activate or not)
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    
    def __init__(self, kernel=(1, 1), stride=(1, 1), r=16, **kwargs):
        super(MetaAconC, self).__init__(**kwargs)
        self.kernel = kernel
        self.stride = stride
        self.r = r
        
    def build(self, input_shape):
        out_dim1 = input_shape[-1]
        out_dim2 = max(self.r, out_dim1 // self.r)
        self.p1 = self.add_weight(
            'aconc/p1',
            shape       = (1, 1, 1, out_dim1),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.p2 = self.add_weight(
            'aconc/p2',
            shape       = (1, 1, 1, out_dim1),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.conv1 = Conv2D(filters=out_dim2, 
                            kernel_size=self.kernel, 
                            strides=(1, 1),
                            padding="valid",
                            use_bias=True)
        self.conv2 = Conv2D(filters=out_dim1, 
                            kernel_size=self.kernel, 
                            strides=(1, 1),
                            padding="valid",
                            use_bias=True)
        
    def call(self, inputs, training=False):
        y = tf.reduce_mean(inputs, axis=1, keepdims=True)
        y = tf.reduce_mean(y, axis=2, keepdims=True)
        beta = self.conv1(y, training=training)
        beta = self.conv2(beta, training=training)
        beta = tf.keras.backend.sigmoid(beta)
        dpx = (self.p1 - self.p2) * inputs
        return dpx * tf.keras.backend.sigmoid(beta * dpx) + self.p2 * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
                "kernel": self.kernel,
                "stride": self.stride,
                "r": self.r
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)