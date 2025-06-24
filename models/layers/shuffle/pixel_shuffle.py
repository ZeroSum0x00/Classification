import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upscale_factor=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscale_factor = upscale_factor 

    def call(self, inputs, training=False):
        N, H, W, C = tf.shape(inputs)
        S = tf.cast(self.upscale_factor , dtype=tf.int32)
        x = tf.reshape(inputs, shape=(N, S, H // S, S, W // S, C))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, shape=(N, H * S, W * S, -1))
        return x
    