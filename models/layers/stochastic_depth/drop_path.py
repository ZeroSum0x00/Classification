import tensorflow as tf
from tensorflow.keras.layers import Dropout



class DropPathV1(tf.keras.layers.Layer):
    """
    Stochastic Depth / DropPath layer
    Reference: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=False):
        if (not training) or (self.drop_prob == 0.0):
            return inputs

        keep_prob = 1.0 - self.drop_prob

        # Generate binary mask with shape [batch_size, 1, 1, 1, ...]
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        shape = (batch_size,) + (1,) * (len(inputs.shape) - 1)

        # Random tensor and binary mask
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)

        # Apply drop path
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output



class DropPathV2(tf.keras.layers.Layer):
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
            inputs = self.drop_layer(inputs, training=training)
        return inputs
    