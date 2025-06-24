import tensorflow as tf



class ClassToken(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.class_tokens = self.add_weight(
            name="tokens",
            shape=(1, 1, dim),
             initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        class_tokens = tf.repeat(self.class_tokens, tf.shape(inputs)[0], axis=0)
        return tf.concat([class_tokens, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


class ClassificationToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]

        self.cls = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer=tf.initializers.zeros(),
            trainable=True,
            name="cls_variable"
        )
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size])
        cls_broadcasted = tf.cast(cls_broadcasted, dtype=inputs.dtype)
        return tf.concat([cls_broadcasted, inputs], axis=1)
    