import numpy as np
import tensorflow as tf


class PositionalIndex(tf.keras.layers.Layer):
    def __init__(self, block_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.use_layer_as_module = True

    def build(self, input_shape):
        pos_idx = np.arange(0, self.block_size, dtype="int64").reshape(1, -1)
        self.pos_idx = tf.convert_to_tensor(pos_idx, dtype="int64")
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.pos_idx[:, : inputs.shape[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    