import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class ChannelAffine(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = tf.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(
            name="weight",
            shape=ww_shape,
            initializer=self.ww_init,
            trainable=True,
        )
        
        if self.use_bias:
            self.bb = self.add_weight(
                name="bias",
                shape=ww_shape,
                initializer=self.bb_init,
                trainable=True,
            )
            
        super().build(input_shape)

    def call(self, inputs, training=False):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        if K.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.squeeze(ii) for ii in weights]
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        if K.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.reshape(ii, self.ww.shape) for ii in weights]
        return self.set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config
