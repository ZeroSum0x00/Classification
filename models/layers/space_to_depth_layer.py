import tensorflow as tf




class SpaceToDepthV1(tf.keras.layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, block_size=self.block_size)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        return (batch,
                h // self.block_size,
                w // self.block_size,
                c * self.block_size ** 2)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

        
class SpaceToDepthV2(tf.keras.layers.Layer):
    def __init__(self, block_size=2, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h, w, c = tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        bs = self.block_size

        x = tf.reshape(inputs, (batch_size,
                                h // bs, bs,
                                w // bs, bs,
                                c))
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        output = tf.reshape(x, (batch_size,
                                h // bs,
                                w // bs,
                                c * bs * bs))
        return output

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        bs = self.block_size
        return (batch,
                h // bs if h is not None else None,
                w // bs if w is not None else None,
                c * bs * bs if c is not None else None)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config
        