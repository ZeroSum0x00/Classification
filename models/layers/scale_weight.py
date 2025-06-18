import tensorflow as tf
import tensorflow.keras.backend as K


class ScaleWeight(tf.keras.layers.Layer):
    def __init__(self, scale_ratio=1.0, use_bias=True, *args, **kwargs):
        super(ScaleWeight, self).__init__(*args, **kwargs)
        self.scale_ratio = scale_ratio
        self.use_bias = use_bias
        
    def build(self, input_shape):
        self.weight_ratio = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(value=self.scale_ratio),
            trainable=True,
            name="weight_ratio"
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_shape[-1],),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                name="bias"
            )
        else:
            self.bias = None
            
        super(ScaleWeight, self).build(input_shape)

    def call(self, inputs, training=False):
        output = inputs * self.weight_ratio
        if self.bias is not None:
            output += self.bias
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale_ratio": self.scale_ratio,
            "use_bias": self.use_bias
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    