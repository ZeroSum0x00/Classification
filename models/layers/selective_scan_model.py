import tensorflow as tf
from tensorflow.keras.layers import Dense
from utils.model_processing import check_regularizer
from models.layers import get_activation_from_name, get_normalizer_from_name



class SSM(tf.keras.layers.Layer):

    """
    State Space Models (SSM) uses selective scan algorithm

    args:
      dt_rank (int): The rank of the state space model.
      dim_inner (int): The dimension of the inner layer of the multi-head attention.
      d_state (int): The dimension of the state space model.
      activation (str): activation name.

    returns:
      output: result of the ssm

    """

    def __init__(
        self,
        dt_rank,
        dim_inner,
        d_state,
        activation="silu",
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.deltaBC_layer = Dense(
            units=self.dt_rank + 2 * self.d_state,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.dt_proj_layer = Dense(
            units=self.dim_inner,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )

        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activation = get_activation_from_name(self.activation)
        
        with tf.init_scope():
            A_value = tf.range(1, self.d_state + 1, delta=1, dtype=tf.float32)
            A_value = tf.expand_dims(A_value, axis=0)
            A_value = tf.repeat(A_value, repeats=256, axis=0)
            A_value = tf.math.log(A_value)
            
        self.A_log = self.add_weight(
            shape=A_value.shape,
            initializer=A_value,
            trainable=True,
            name="A_log"
        )
        
        self.D = self.add_weight(
            shape=(self.dim_inner,),
            initializer=tf.initializers.ones(),
            trainable=True,
            name="D"
        )

    def selective_scan(self,input, delta, A, B, C, D):
        """
        Calculate output of the selective state space model using parallel scan
        implemented using the cumulative sum

        args:
          input: data input that we calculate the ssm on.
          delta: mediates how much focus is put on new input.
          A: state matrix controlling the hidden state.
          B: modulate the recurrent dynamics based on content (input).
          C: modulate the recurrent dynamics based on context (hidden states).
          D: scales the skip connection.

        returns:
          output: result of the ssm with current parameters

        """
        # first step of discretization of A
        deltaA = tf.einsum("bld,dn->bldn", delta, A) # quasi delta mal A
        deltaBinput = tf.einsum("bld,bld,bln->bldn", delta, input, B) # input mal B mal delta

        deltaA_cumsum = tf.pad(
            deltaA[:, 1:],
            paddings=
            [
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 0]
            ],
        )[:, 1:, :, :]

        deltaA_cumsum = tf.reverse(deltaA_cumsum, axis=[1])  # Flip along axis 1

        # Cumulative sum along all the input tokens, parallel prefix sum,
        # calculates dA for all the input tokens in parallel
        deltaA_cumsum = tf.math.cumsum(deltaA_cumsum, axis=1)

        # second step of discretization of A
        deltaA_cumsum = tf.exp(deltaA_cumsum)
        deltaA_cumsum = tf.reverse(deltaA_cumsum, axis=[1])  # Flip back along axis 1

        # calculate intermediate output as in graphs shown for ssm"s
        x = deltaBinput * deltaA_cumsum
        # 1e-12 to avoid division by 0
        x = tf.math.cumsum(x, axis=1) / (deltaA_cumsum + 1e-12)

        # intermediate output multiplied with parameter C
        output = tf.einsum("bldn,bln->bld", x, C)

        return output + input * D

    def call(self, inputs, training=False):
        A = -tf.math.exp(self.A_log)
        D = self.D
        deltaBC = self.deltaBC_layer(inputs, training=training)
        deltaBC = self.norm_layer1(deltaBC, training=training)
        deltaBC = self.activation(deltaBC, training=training)
        
        delta, B, C = tf.split(
            deltaBC,
            num_or_size_splits=[self.dt_rank, self.d_state, self.d_state],
            axis=-1,
        )

        delta = self.dt_proj_layer(delta, training=training)
        delta = self.norm_layer2(delta, training=training)
        delta = tf.nn.softplus(delta)

        x = self.selective_scan(inputs, delta, A, B, C, D)
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "dt_rank": self.dt_rank,
            "dim_inner": self.dim_inner,
            "d_state": self.d_state,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)