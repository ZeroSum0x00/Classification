import tensorflow as tf
from tensorflow.keras.layers import Dropout


class StochasticDepth(tf.keras.layers.Layer):
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

    def call(self, x, training=None):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x
        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_l = tf.keras.backend.random_bernoulli(
            [], p=self.survival_probability, dtype=self._compute_dtype_object
        )

        def _call_train():
            return shortcut + b_l * residual

        def _call_test():
            return shortcut + self.survival_probability * residual

        return tf.keras.backend.in_train_phase(
            _call_train, _call_test, training=training
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"survival_probability": self.survival_probability}
        return {**base_config, **config}


class StochasticDepth2(tf.keras.layers.Layer):
  
    def __init__(self, survival_probability: float=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.survival_probability = survival_probability
    
    def __drop_block(self, inputs, drop_rate=0):
        """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
        if drop_rate > 0:
            noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
            return Dropout(drop_rate, noise_shape=noise_shape)(inputs)
        else:
            return inputs

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x
        x = shortcut + self.__drop_block(residual, self.survival_probability)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()

        config = {"survival_probability": self.survival_probability}

        return {**base_config, **config}
    