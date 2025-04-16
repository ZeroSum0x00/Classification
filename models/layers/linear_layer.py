import tensorflow as tf
import tensorflow.keras.backend as K


class LinearLayer(tf.keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super(LinearLayer, self).__init__(name=name)
        
    def call(self, inputs, training=False):
        return inputs
