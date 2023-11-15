import tensorflow as tf
import tensorflow.keras.backend as K


class LinearLayer(tf.keras.layers.Layer):

    def call(self, inputs, training=False):
        return inputs