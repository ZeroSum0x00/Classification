import tensorflow as tf



class ChannelAffine(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1.0, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.weight_init_value = weight_init_value
        self.axis = axis
        self.supports_masking = False

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        rank = len(input_shape)

        if isinstance(self.axis, int):
            axis = [self.axis]
        else:
            axis = list(self.axis)

        # Normalize negative axes
        axis = [a if a >= 0 else rank + a for a in axis]

        # Create shape for weights
        ww_shape = [1] * rank
        for a in axis:
            ww_shape[a] = input_shape[a]

        # Use correct initializer
        if self.weight_init_value == 1:
            ww_initializer = tf.keras.initializers.Ones()
        else:
            ww_initializer = tf.keras.initializers.Constant(self.weight_init_value)

        self.ww = self.add_weight(
            shape=ww_shape,
            initializer=ww_initializer,
            trainable=True,
            name="weight"
        )

        if self.use_bias:
            self.bb = self.add_weight(
                shape=ww_shape,
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                name="bias"
            )
        else:
            self.bb = None

        super().build(input_shape)

    def call(self, inputs, training=False):
        output = inputs * self.ww
        if self.bb is not None:
            output += self.bb
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_bias": self.use_bias,
            "weight_init_value": self.weight_init_value,
            "axis": self.axis,
        })
        return config
