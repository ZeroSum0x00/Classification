import numpy as np
import tensorflow as tf



class CausalMask(tf.keras.layers.Layer):
    def __init__(self, block_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.use_layer_as_module = True

    def build(self, input_shape):
        causal_mask = (1 - np.tri(self.block_size).astype("float32")[None, None]) * -65504
        self.causal_mask = tf.convert_to_tensor(causal_mask, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, training=False):
        return inputs + self.causal_mask[:, :, : inputs.shape[2], : inputs.shape[3]]

    def get_config(self):
        config = super().get_config()
        config.update({
            "block_size": self.block_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    