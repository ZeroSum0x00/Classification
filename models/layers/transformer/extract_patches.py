import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D



class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, patch_size, lasted_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.lasted_dim = lasted_dim

    def build(self, input_shape):
        self.hidden_dim = np.prod(input_shape[1:]) // self.lasted_dim
        
        self.extractor = Conv2D(
            filters=self.lasted_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
        )
                
    def call(self, inputs, training=False):
        x = self.extractor(inputs, training=training)
        x = tf.reshape(x, shape=[-1, self.hidden_dim, self.lasted_dim])
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "hidden_dim": self.hidden_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
