import tensorflow as tf
import tensorflow.keras.backend as K


class LinearLayer(tf.keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def call(self, inputs, training=False):
        return inputs
