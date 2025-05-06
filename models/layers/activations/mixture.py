import tensorflow as tf


class Mixture(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Mixture, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.p = self.add_weight(
            "mixture/p",
            shape=(1),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs, training=False):
        return self.p * inputs + (1 - self.p) * tf.nn.relu(inputs)
    