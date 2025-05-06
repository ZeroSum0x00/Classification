import tensorflow as tf
from tensorflow.keras.layers import Dense


class SSM(tf.keras.layers.Layer):

    """
    State Space Models (SSM) uses selective scan algorithm

    args:
      dt_rank (int): The rank of the state space model.
      dim_inner (int): The dimension of the inner layer of the multi-head attention.
      d_state (int): The dimension of the state space model.
      activation (str): activation name.
      normalizer (str): normalization name.

    returns:
      output: result of the ssm

    """

    def __init__(
        self,
        dt_rank,
        dim_inner,
        d_state,
        activation="relu",
        normalizer="batch-norm",
        *args, **kwargs
    ):
        super(SSM, self).__init__(*args, **kwargs)
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        self.deltaBC_layer = Dense(self.dt_rank + 2 * self.d_state, use_bias=False)
        self.dt_proj_layer = Dense(self.dim_inner)

        with tf.init_scope():
            A_value = tf.range(1, self.d_state + 1, delta=1, dtype=tf.float32)
            A_value = tf.expand_dims(A_value, axis=0)
            A_value = tf.repeat(A_value, repeats=256, axis=0)
            A_value = tf.math.log(A_value)
            
        self.A_log = tf.Variable(
            initial_value=A_value,
            trainable=True,
            name="ssm.A_log"
        )
        
        self.D = self.add_weight(
            name="ssm.D",
            shape=(self.dim_inner,),
            initializer=tf.initializers.ones(),
            trainable=True
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
            deltaA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

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
        x = tf.math.cumsum(x, axis=1)/(deltaA_cumsum + 1e-12)

        # intermediate output multiplied with parameter C
        output = tf.einsum("bldn,bln->bld", x, C)

        return output + input * D

    def call(self, inputs, training=False):
        A = -tf.math.exp(self.A_log)
        D = self.D
        deltaBC = self.deltaBC_layer(inputs, training=training)
        
        delta, B, C = tf.split(
            deltaBC,
            num_or_size_splits=[self.dt_rank, self.d_state, self.d_state],
            axis=-1,
        )

        delta = self.dt_proj_layer(delta, training=training)
        delta = tf.nn.softplus(delta)

        x = self.selective_scan(inputs, delta, A, B, C, D)
        return x
    