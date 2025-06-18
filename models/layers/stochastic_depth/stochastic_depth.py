import tensorflow as tf
from tensorflow.keras.layers import Dropout


class StochasticDepthV1(tf.keras.layers.Layer):
    """Stochastic Depth layer.
    Implements Stochastic Depth as described in
    [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), to randomly drop residual branches
    in residual architectures.
    Usage:
    Residual architectures with fixed depth, use residual branches that are merged back into the main network
    by adding the residual branch back to the input:
    >>> input = np.ones((1, 3, 3, 1), dtype = np.float32)
    >>> residual = tf.keras.layers.Conv2D(1, 1)(input)
    >>> output = tf.keras.layers.Add()([input, residual])
    >>> output.shape
    TensorShape([1, 3, 3, 1])
    StochasticDepth acts as a drop-in replacement for the addition:
    >>> input = np.ones((1, 3, 3, 1), dtype = np.float32)
    >>> residual = tf.keras.layers.Conv2D(1, 1)(input)
    >>> output = tfa.layers.StochasticDepth()([input, residual])
    >>> output.shape
    TensorShape([1, 3, 3, 1])
    At train time, StochasticDepth returns:
    $$
    x[0] + b_l * x[1],
    $$
    where $b_l$ is a random Bernoulli variable with probability $P(b_l = 1) = p_l$
    At test time, StochasticDepth rescales the activations of the residual branch based on the survival probability ($p_l$):
    $$
    x[0] + p_l * x[1]
    $$
    Args:
        survival_probability: float, the probability of the residual branch being kept.
    Call Args:
        inputs:  List of `[shortcut, residual]` where `shortcut`, and `residual` are tensors of equal shape.
    Output shape:
        Equal to the shape of inputs `shortcut`, and `residual`
    """
    def __init__(self, survival_probability: float=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.survival_probability = survival_probability

    def call(self, inputs, training=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Input must be a list of length 2: [shortcut, residual]")

        shortcut, residual = inputs
        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_l = tf.keras.backend.random_bernoulli(
            [], p=self.survival_probability, dtype=tf.float32
        )

        if training:
            return shortcut + b_l * residual
        else:
            return shortcut + self.survival_probability * residual

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "survival_probability": self.survival_probability
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StochasticDepthV2(tf.keras.layers.Layer):
    def __init__(self, survival_probability: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.survival_probability = survival_probability

    def call(self, inputs, training=None):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Input must be a list of length 2: [shortcut, residual]")
            
        shortcut, residual = inputs

        if training:
            batch_size = tf.shape(residual)[0]
            shape = (batch_size,) + (1,) * (len(residual.shape) - 1)
            random_tensor = self.survival_probability + tf.random.uniform(shape, dtype=residual.dtype)
            binary_tensor = tf.floor(random_tensor)
            residual = residual / self.survival_probability * binary_tensor
        else:
            residual = residual  # or optionally: residual *= self.survival_probability

        return shortcut + residual

    def get_config(self):
        config = super().get_config()
        config.update({
            "survival_probability": self.survival_probability
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        