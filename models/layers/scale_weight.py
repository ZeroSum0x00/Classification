import tensorflow as tf
import tensorflow.keras.backend as K


class ScaleWeight(tf.keras.layers.Layer):
    def __init__(self, scale_ratio=1.0, use_bias=True, *args, **kwargs):
        super(ScaleWeight, self).__init__(*args, **kwargs)
        self.scale_ratio = scale_ratio
        self.use_bias = use_bias
        
    def build(self, input_shape):
        weights_init = tf.keras.initializers.Constant(value=self.scale_ratio)
        self.weight_ratio = tf.Variable(
            initial_value=weights_init(shape=(input_shape[-1])),
            trainable=True,
            name=f"scale-weight/weights"
        )
        if self.use_bias:
            bias_init = tf.keras.initializers.Zeros()
            self.bias = tf.Variable(
            initial_value=bias_init(shape=(input_shape[-1])),
            trainable=True,
            name=f"scale-weight/bias"
        )
        super(ScaleWeight, self).build(input_shape)

    def call(self, inputs, training=False):
        if hasattr(self, "bias"):
            return inputs * self.weight_ratio + self.bias
        else:
            return inputs * self.weight_ratio

    def get_config(self):
        config = super(ScaleWeight, self).get_config()
        config.update({
            "scale_ratio": self.scale_ratio,
            "use_bias": self.use_bias
        })
        return config
    