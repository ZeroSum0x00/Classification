import tensorflow as tf


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, upscale_factor=4, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.upscale_factor  = upscale_factor 
        
    def call(self, inputs, training=False):
        N, H, W, C = inputs.shape
        S = C // self.upscale_factor
        Z = self.upscale_factor
        x = tf.reshape(inputs, shape=(-1, H, W, S, Z))
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, shape=(-1, H, W, C))
        return x