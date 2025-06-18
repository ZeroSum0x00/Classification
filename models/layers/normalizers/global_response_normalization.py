import tensorflow as tf
from models.layers import ChannelAffine


class GlobalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.axis = axis

    def build(self, input_shape):
        actual_axis = (len(input_shape) + self.axis) if self.axis < 0 else self.axis
        
        self.affine = ChannelAffine(
            use_bias=True,
            weight_init_value=0,
            axis=actual_axis,
            name="channel_affine"
        )
        
        self.affine.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        axis = (len(inputs.shape) + self.axis) if self.axis < 0 else self.axis
        norm_axes = [i for i in range(1, len(inputs.shape)) if i != axis]

        norm_scale = tf.shape(inputs)[1] * tf.shape(inputs)[2]
        norm_scale = tf.cast(norm_scale, inputs.dtype) ** 0.5

        nn = tf.reduce_mean(tf.square(inputs), axis=norm_axes, keepdims=True)
        nn = tf.sqrt(nn) * norm_scale
        nn = nn / (tf.reduce_mean(nn, axis=axis, keepdims=True) + self.epsilon)

        x_normed = inputs * nn
        x_normed = self.affine(x_normed)
        return x_normed + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config
