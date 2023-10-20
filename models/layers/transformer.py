import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from . import get_activation_from_name, get_normalizer_from_name


@tf.keras.utils.register_keras_serializable()
class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_dim, *args, **kwargs):
        super(ExtractPatches, self).__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.extractor = Conv2D(filters=self.hidden_dim,
                                kernel_size=self.patch_size,
                                strides=self.patch_size,
                                padding="valid",
                                name="embedding")
        self.reshape = Reshape((-1, self.hidden_dim))
        
    def call(self, inputs):
        x = self.extractor(inputs)
        x = self.reshape(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ClassificationToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(name="cls_variable",
                          initial_value=cls_init(shape=(1, 1, input_shape[-1]),
                                                 dtype=tf.float32),
                          trainable=True)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
                                  dtype=inputs.dtype)
        return tf.concat([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DistillationToken(tf.keras.layers.Layer):
    """Append a distillation token to an input layer."""

    def build(self, input_shape):
        dist_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.dist = tf.Variable(name="dist_variable",
                                initial_value=dist_init(shape=(1, 1, input_shape[-1]),
                                dtype=tf.float32),
                                trainable=True)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        dist_broadcasted = tf.cast(tf.broadcast_to(self.dist, [batch_size, 1, self.hidden_size]),
                                   dtype=inputs.dtype)
        return tf.concat([dist_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbedding(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Number of dimensions should be 3, got {len(input_shape)}")

        pe_init = tf.random_normal_initializer(stddev=0.06)
        self.pos_embedding = tf.Variable(name="pos_embedding",
                                         initial_value=pe_init(shape=(1, input_shape[1], input_shape[2])),
                                         dtype=tf.float32,
                                         trainable=True)

    def call(self, inputs):
        return inputs + tf.cast(self.pos_embedding, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    "Link: https://arxiv.org/pdf/1706.03762.pdf"
    
    def __init__(self, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = Dense(hidden_size, name="query")
        self.key_dense = Dense(hidden_size, name="key")
        self.value_dense = Dense(hidden_size, name="value")
        self.combine_heads = Dense(hidden_size, name="out")

    def scaled_dot_product_attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.key_dense(inputs)
        key = self.separate_heads(key, batch_size)
        value = self.value_dense(inputs)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.scaled_dot_product_attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_heads": self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=-1, use_conv=False, use_bias=True, use_gated=False, activation='gelu', normalizer=None, drop_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_dim    = mlp_dim
        self.out_dim    = out_dim
        self.use_conv   = use_conv
        self.use_bias   = use_bias
        self.use_gated  = use_gated
        self.activation = activation
        self.normalizer = normalizer
        self.drop_rate  = drop_rate
    
    def build(self, input_shape):
        hidden_dim = self.mlp_dim * 2 if self.use_gated else self.mlp_dim
        if not self.use_conv:
            self.linear1 = Dense(hidden_dim, use_bias=self.use_bias)
            self.linear2 = Dense(self.out_dim if self.out_dim > 0 else input_shape[-1], use_bias=self.use_bias)
        else:
            self.linear1 = Conv2D(filters=hidden_dim, 
                                  kernel_size=(1, 1), 
                                  strides=(1, 1),
                                  use_bias=self.use_bias)
            self.linear2 = Conv2D(self.out_dim if self.out_dim > 0 else input_shape[-1],
                                  kernel_size=(1, 1), 
                                  strides=(1, 1),
                                  use_bias=self.use_bias)
        if self.normalizer:
            self.norm = get_normalizer_from_name(self.normalizer)
            
        self.activation = get_activation_from_name(self.activation)
        self.dropout = Dropout(self.drop_rate)
        
    def call(self, inputs, training):
        x = self.linear1(inputs, training=training)
        
        if self.use_gated:
            gate, x = tf.split(x, 2, axis=-1)
            gate = self.activation(gate)
            x = gate * x
        else:
            x = self.activation(x)
            
        x = self.dropout(x)
        
        if self.normalizer:
            x = self.norm(x, training=training)
        x = self.linear2(x, training=training)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
                "mlp_dim": self.mlp_dim,
                "out_dim": self.out_dim,
                "use_conv": self.use_conv,
                "use_bias": self.use_bias,
                "use_gated": self.use_gated,
                "activation": self.activation,
                "normalizer": self.normalizer,
                "drop_rate": self.drop_rate,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    "Link: https://arxiv.org/pdf/1706.03762.pdf"

    def __init__(self, num_heads, mlp_dim, normalizer='batch-norm', norm_eps=1e-6, drop_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.normalizer = normalizer
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        self.attention = MultiHeadSelfAttention(num_heads=self.num_heads,
                                                name="MultiHeadDotProductAttention_1")
        self.mlpblock = MLPBlock(self.mlp_dim, drop_rate=self.drop_rate, name="MlpBlock")
        self.layernorm1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps, name="LayerNorm_0")
        self.layernorm2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps, name="LayerNorm_2")
        self.dropout_layer = Dropout(self.drop_rate)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs)
        x, weights = self.attention(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "normalizer": self.normalizer,
                "norm_eps": self.norm_eps,
                "drop_rate": self.drop_rate
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
