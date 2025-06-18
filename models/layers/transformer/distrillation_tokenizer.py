import tensorflow as tf



class DistillationToken(tf.keras.layers.Layer):
    """Append a distillation token to an input layer."""

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]

        self.dist = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer=tf.initializers.zeros(),
            trainable=True,
            name="dist_variable"
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        dist_broadcasted = tf.broadcast_to(self.dist, [batch_size, 1, self.hidden_size])
        dist_broadcasted = tf.cast(dist_broadcasted, dtype=inputs.dtype)
        return tf.concat([dist_broadcasted, inputs], axis=1)
    