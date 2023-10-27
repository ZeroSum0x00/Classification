import tensorflow as tf
from tensorflow.keras.layers import Dropout


# def drop_path(inputs, drop_prob, is_training):
#     if (not is_training) or (drop_prob == 0.):
#         return inputs

#     # Compute keep_prob
#     keep_prob       = 1.0 - drop_prob

#     # Compute drop_connect tensor
#     random_tensor   = keep_prob
#     shape           = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
#     random_tensor   += tf.random.uniform(shape, dtype=inputs.dtype)
#     binary_tensor   = tf.floor(random_tensor)
#     output          = tf.math.divide(inputs, keep_prob) * binary_tensor
#     return output


# class DropPath(tf.keras.layers.Layer):
#     """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    
#     def __init__(self, drop_prob=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.drop_prob = drop_prob

#     def call(self, x, training=False):
#         return drop_path(x, self.drop_prob, training)


class DropPath(tf.keras.layers.Layer):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    
    def __init__(self, drop_prob=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_prob = drop_prob

    def build(self, input_shape):
        if self.drop_prob > 0:
            noise_shape = [None] + [1] * (len(input_shape) - 1)  # [None, 1, 1, 1]
            self.drop_layer = Dropout(self.drop_prob, noise_shape=noise_shape)
        else:
            self.drop_layer = None
            
    def call(self, inputs, training=False):
        if training and self.drop_layer is not None:
            inputs = self.drop_layer(inputs)
        return inputs