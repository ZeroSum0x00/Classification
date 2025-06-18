import tensorflow as tf



class PositionalEmbedding(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Number of dimensions should be 3, got {len(input_shape)}")
        
        self.pos_embedding = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
            trainable=True,
            name="pos_embedding"
        )

    def call(self, inputs, training=False):
        return inputs + tf.cast(self.pos_embedding, dtype=inputs.dtype)
        