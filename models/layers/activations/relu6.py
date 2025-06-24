import tensorflow as tf
from tensorflow.keras.layers import ReLU



class ReLU6(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.activation = ReLU(max_value=6.)
        
    def call(self, inputs, training=False):
        return self.activation(inputs)
    