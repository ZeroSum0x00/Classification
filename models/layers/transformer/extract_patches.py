import tensorflow as tf
from tensorflow.keras.layers import Conv2D



class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, patch_size, lasted_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.lasted_dim = lasted_dim

    def build(self, input_shape):
        self.extractor = Conv2D(
            filters=self.lasted_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
        )
                
    def call(self, inputs, training=False):
        x = self.extractor(inputs, training=training)
        batch_size = tf.shape(x)[0]
        num_patches = x.shape[1] * x.shape[2]
        x = tf.reshape(x, shape=[batch_size, num_patches, self.lasted_dim])
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "lasted_dim": self.lasted_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
