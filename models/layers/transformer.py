import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import add
from . import get_activation_from_name, get_normalizer_from_name
from .channel_affine import ChannelAffine
from .single_stochastic_depth import DropPath
from .add_bias import BiasLayer


@tf.keras.utils.register_keras_serializable()
class CausalMask(tf.keras.layers.Layer):
    def __init__(self, block_size, *args, **kwargs):
        super(CausalMask, self).__init__(*args, **kwargs)
        self.block_size = block_size
        self.use_layer_as_module = True

    def build(self, input_shape):
        causal_mask = (1 - np.tri(self.block_size).astype("float32")[None, None]) * -65504
        self.causal_mask = tf.convert_to_tensor(causal_mask, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, training=False):
        return inputs + self.causal_mask[:, :, : inputs.shape[2], : inputs.shape[3]]

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config


@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(ClassToken, self).__init__(*args, **kwargs)
        self.token_init = tf.keras.initializers.TruncatedNormal(stddev=0.2)

    def build(self, input_shape):
        self.class_tokens = self.add_weight(name="tokens", 
                                            shape=(1, 1, input_shape[-1]), 
                                            initializer=self.token_init, 
                                            trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=False):
        class_tokens = tf.repeat(self.class_tokens, tf.shape(inputs)[0], axis=0)
        return tf.concat([class_tokens, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


@tf.keras.utils.register_keras_serializable()
class PositionalIndex(tf.keras.layers.Layer):
    def __init__(self, block_size=1024, *args, **kwargs):
        super(PositionalIndex, self).__init__(*args, **kwargs)
        self.block_size = block_size
        self.use_layer_as_module = True

    def build(self, input_shape):
        pos_idx = np.arange(0, self.block_size, dtype="int64").reshape(1, -1)
        self.pos_idx = tf.convert_to_tensor(pos_idx, dtype="int64")
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.pos_idx[:, : inputs.shape[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

        
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
        
    def call(self, inputs, training=False):
        x = self.extractor(inputs, training=training)
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
                               initial_value=cls_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
                               trainable=True)
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]), dtype=inputs.dtype)
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
    
    def call(self, inputs, training=False):
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
class PositionalEmbedding(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Number of dimensions should be 3, got {len(input_shape)}")

        pe_init = tf.random_normal_initializer(stddev=0.06)
        self.pos_embedding = tf.Variable(name="pos_embedding",
                                         initial_value=pe_init(shape=(1, input_shape[1], input_shape[2])),
                                         dtype=tf.float32,
                                         trainable=True)

    def call(self, inputs, training=False):
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
    
    def __init__(self, num_heads, return_weight=False, *args, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(*args, **kwargs)
        self.num_heads     = num_heads
        self.return_weight = return_weight
        
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

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs, training=training)
        query = self.separate_heads(query, batch_size)
        key = self.key_dense(inputs, training=training)
        key = self.separate_heads(key, batch_size)
        value = self.value_dense(inputs, training=training)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.scaled_dot_product_attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention, training=training)
        if self.return_weight:
            return output, weights
        else:
            return output, None

    def get_config(self):
        config = super().get_config()
        config.update({
                "num_heads": self.num_heads,
                "return_weight": self.return_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=-1, use_conv=False, use_bias=True, use_gated=False, activation='gelu', normalizer=None, drop_rate=0.1, *args, **kwargs):
        super(MLPBlock, self).__init__(*args, **kwargs)
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
        
    def call(self, inputs, training=False):
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
class AttentionMLPBlock(tf.keras.layers.Layer):
    
    def __init__(self, 
                 attention_layer,
                 mlp_ratio=4, 
                 layer_scale=0.1, 
                 use_gated_mlp=False,
                 activation='gelu',
                 normalizer='layer-norm',
                 use_mlp_norm=False, 
                 norm_eps=1e-6,
                 drop_rate=0.1,
                 drop_prob=0.0, 
                 *args, 
                 **kwargs):
        super(AttentionMLPBlock, self).__init__(*args, **kwargs)
        self.attention_layer = attention_layer
        self.mlp_ratio       = mlp_ratio
        self.layer_scale     = layer_scale
        self.use_gated_mlp   = use_gated_mlp
        self.activation      = activation
        self.normalizer      = normalizer
        self.mlp_normalizer  = normalizer if use_mlp_norm else None
        self.norm_eps        = norm_eps
        self.drop_rate       = drop_rate
        self.drop_prob       = drop_prob

    def build(self, input_shape):
        self.mlp_block   = MLPBlock(mlp_dim=int(input_shape[-1] * self.mlp_ratio), 
                                    use_gated=self.use_gated_mlp, 
                                    activation=self.activation, 
                                    normalizer=self.mlp_normalizer, 
                                    drop_rate=self.drop_rate)
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        
        if self.layer_scale > 0:
            self.affine1 = ChannelAffine(use_bias=False, weight_init_value=self.layer_scale)
            self.affine2 = ChannelAffine(use_bias=False, weight_init_value=self.layer_scale)
            
        self.drop1 = DropPath(drop_prob=self.drop_prob)
        self.drop2 = DropPath(drop_prob=self.drop_prob)

    def call(self, inputs, training=False):
        x = self.norm_layer1(inputs, training=training)
        x = self.attention_layer(x, training=training)
        if self.layer_scale > 0:
            x = self.affine1(x, training=training)
        x = self.drop1(x, training=training)
        attn_out = add([inputs, x])

        x = self.norm_layer2(attn_out, training=training)
        x = self.mlp_block(x, training=training)
        if self.layer_scale > 0:
            x = self.affine2(x, training=training)
        x = self.drop2(x, training=training)
        x = add([attn_out, x])
        return x

        
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    "Link: https://arxiv.org/pdf/1706.03762.pdf"

    def __init__(self, num_heads, mlp_dim, return_weight=False, activation='gelu', normalizer=None, norm_eps=1e-6, drop_rate=0.1, *args, **kwargs):
        super(TransformerBlock, self).__init__(*args, **kwargs)
        self.num_heads     = num_heads
        self.mlp_dim       = mlp_dim
        self.return_weight = return_weight
        self.activation    = activation
        self.normalizer    = normalizer
        self.norm_eps      = norm_eps
        self.drop_rate     = drop_rate
    
    def build(self, input_shape):
        self.attention = MultiHeadSelfAttention(num_heads=self.num_heads,
                                                return_weight=self.return_weight,
                                                name="MultiHeadDotProductAttention_1")
        self.mlpblock = MLPBlock(self.mlp_dim,
                                 activation=self.activation,
                                 normalizer=self.normalizer, 
                                 drop_rate=self.drop_rate, 
                                 name="MlpBlock")
        self.layernorm1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps, name="LayerNorm_0")
        self.layernorm2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps, name="LayerNorm_2")
        self.dropout_layer = Dropout(self.drop_rate)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs, training=training)
        x, weights = self.attention(x, training=training)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x, training=training)
        y = self.mlpblock(y, training=training)
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


@tf.keras.utils.register_keras_serializable()
class PositionalEncodingFourierRot1D(tf.keras.layers.Layer):
    def __init__(self, max_block_size, temperature=1e4, *args, **kwargs):
        super(PositionalEncodingFourierRot1D, self).__init__(*args, **kwargs)
        self.max_block_size = max_block_size
        self.temperature    = float(temperature)

    def build(self, input_shape):
        self.channels = input_shape[-2] * input_shape[-1]
        pos_filters = self.channels // 2
        dim_t = self.temperature ** (np.arange(pos_filters, dtype="float32") / pos_filters)
        grid = np.expand_dims(np.arange(self.max_block_size, dtype="float32"), -1) / dim_t
        pos_sin, pos_cos = np.expand_dims(np.sin(grid), -2), np.expand_dims(np.cos(grid), -2)
        self.pos_sin = tf.convert_to_tensor(pos_sin, dtype=self.compute_dtype)
        self.pos_cos = tf.convert_to_tensor(pos_cos, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, training=False):
        left, right = tf.unstack(inputs, axis=-1)
        pos_cos = self.pos_cos[: left.shape[-3]]
        pos_sin = self.pos_sin[: left.shape[-3]]
        out = tf.stack([left * pos_cos - right * pos_sin, right * pos_cos + left * pos_sin], axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
                "temperature": self.temperature, 
                "max_block_size": self.max_block_size
        })
        return config


@tf.keras.utils.register_keras_serializable()
class PositionalEncodingFourierRot(tf.keras.layers.Layer):
    def __init__(self, num_heads=-1, attn_height=-1, cls_token=True, temperature=1e4, ref_feature_shape=16, *args, **kwargs):
        super(PositionalEncodingFourierRot, self).__init__(*args, **kwargs)
        self.num_heads         = num_heads
        self.attn_height       = attn_height
        self.cls_token         = cls_token
        self.temperature       = float(temperature)
        self.ref_feature_shape = ref_feature_shape
        self.cls_token_len     = 1 if cls_token else 0

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.attn_height == -1:
            height = width = int(float(input_shape[-2] - self.cls_token_len) ** 0.5)
        else:
            height = self.attn_height
            width = int(float(input_shape[-2] - self.cls_token_len) / height)
        self.blocks_shape = [*input_shape[1:-2], input_shape[-2] - self.cls_token_len]

        hh = np.arange(height, dtype="float32")
        ww = np.arange(width, dtype="float32")
        
        if self.ref_feature_shape is not None and self.ref_feature_shape > 0:
            hh = hh / height * self.ref_feature_shape
            ww = ww / height * self.ref_feature_shape

        pos_fileters = (self.channels // self.num_heads // 4) if self.num_heads > 0 else (self.channels // 4)
        dim_t = self.temperature ** (np.arange(pos_fileters, dtype="float32") / pos_fileters)
        grid = np.stack(np.meshgrid(hh, ww, indexing="ij"), axis=-1)
        grid = np.expand_dims(grid, -1) / dim_t
        grid = np.reshape(grid, [height, width, -1])
        pos_sin, pos_cos = np.sin(grid), np.cos(grid)
        pos_sin, pos_cos = np.repeat(pos_sin, 2, axis=-1), np.repeat(pos_cos, 2, axis=-1)

        if self.num_heads > 0:
            pos_sin = np.repeat(np.expand_dims(pos_sin, axis=-2), self.num_heads, axis=-2).reshape([height * width, self.num_heads * pos_fileters * 4])
            pos_cos = np.repeat(np.expand_dims(pos_cos, axis=-2), self.num_heads, axis=-2).reshape([height * width, self.num_heads * pos_fileters * 4])
        else:
            pos_sin = np.reshape(pos_sin, [height * width, pos_fileters * 4])
            pos_cos = np.reshape(pos_cos, [height * width, pos_fileters * 4])

        self.pos_sin = tf.convert_to_tensor(pos_sin, dtype=self.compute_dtype)
        self.pos_cos = tf.convert_to_tensor(pos_cos, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, training=False):
        if self.cls_token:
            cls_token, inputs = tf.split(inputs, [1, -1], axis=-2)

        left, right = tf.split(tf.reshape(inputs, [-1, *self.blocks_shape, self.channels // 2, 2]), 2, axis=-1)
        rot = tf.concat([-right, left], axis=-1)
        rot = tf.reshape(rot, (-1, *self.blocks_shape, self.channels))
        out = inputs * self.pos_cos + rot * self.pos_sin

        if self.cls_token:
            out = tf.concat([cls_token, out], axis=-2)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
                "attn_height": self.attn_height, 
                "num_heads": self.num_heads,
                "cls_token": self.cls_token, 
                "temperature": self.temperature, 
                "ref_feature_shape": self.ref_feature_shape
        })
        return config


@tf.keras.utils.register_keras_serializable()
class MultiHeadRelativePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_heads=-1, attn_height=-1, cls_token=True, *args, **kwargs):
        super(MultiHeadRelativePositionalEmbedding, self).__init__(*args, **kwargs)
        self.num_heads      = num_heads
        self.attn_height    = attn_height
        self.cls_token      = cls_token

        if cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        if self.attn_height == -1:
            height = width = int(float(attn_shape[2] - self.cls_token_len) ** 0.5)
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
            
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        pos_shape = (num_heads, num_relative_distance)
        self.positional_embedding = self.add_weight(name="positional_embedding", 
                                                    shape=pos_shape, 
                                                    initializer="zeros", 
                                                    trainable=True)

        hh, ww = np.meshgrid(range(height), range(width))
        coords = np.stack([hh, ww], axis=-1)
        coords_flatten = np.reshape(coords, [-1, 2])
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = np.sum(relative_coords, axis=-1).astype("float32")
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.cls_token:
            top = np.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = np.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = np.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            left_corner = np.concatenate([corner, left], axis=0)
            relative_position_index = np.concatenate([top, relative_position_index], axis=0)
            relative_position_index = np.concatenate([left_corner, relative_position_index], axis=1)

        self.relative_position_index = tf.convert_to_tensor(relative_position_index, dtype="int64")
        super().build(attn_shape)

    def call(self, inputs, training=False):
        relative_position_mask = self.relative_position_index[: inputs.shape[2], : inputs.shape[3]]
        pos_emb = tf.gather(self.positional_embedding, relative_position_mask, axis=1)
        return inputs + pos_emb

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
                "attn_height": self.attn_height, 
                "num_heads": self.num_heads,
                "cls_token": self.cls_token
        })
        return base_config

    def load_resized_weights(self, source_layer, method="bilinear"):
        if isinstance(source_layer, dict):
            source_tt = list(source_layer.values())[0]
        else:
            source_tt = source_layer.get_weights()[0]
            
        source_tt = np.array(ssource_tt).astype("float32")
        hh = ww = int(float(source_tt.shape[1] - self.cls_token_pos_len) ** 0.5)
        num_heads = source_tt.shape[0]
        ss = source_tt[:, : hh * ww].reshape((num_heads, hh, ww))

        if self.attn_height == -1:
            target_hh = target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) / target_hh)

        tt = K.numpy_image_resize(ss, target_shape=[target_hh, target_ww], method=method, is_source_channels_last=False)
        tt = tt.reshape((num_heads, tt.shape[1] * tt.shape[2]))
        
        if self.cls_token:
            tt = np.concatenate([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
            
        self.set_weights([tt])

    def show_pos_emb(self, rows=1, base_size=2):
        import math
        import matplotlib.pyplot as plt

        num_heads = self.positional_embedding.shape[0]
        hh = ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        pos_emb = self.positional_embedding[:, : hh * ww]
        pos_emb = pos_emb.numpy() if hasattr(pos_emb, "numpy") else np.array(pos_emb)
        pos_emb = pos_emb.reshape((num_heads, hh, ww))
        cols = int(math.ceil(num_heads / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            if id >= num_heads:
                break
            ax.imshow(pos_emb[id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


@tf.keras.utils.register_keras_serializable()
class EnhanceSelfAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 num_heads, 
                 key_dim=0,
                 attn_height=-1,
                 qk_scale=-1,
                 qv_bias=True,
                 qkv_bias=False,
                 return_weight=True,
                 return_bias=False,
                 pos_emb=False,
                 rotate_pos_emb=False,
                 text_max_block_size=0,
                 attn_dropout=0,
                 *args, 
                 **kwargs):
        super(EnhanceSelfAttention, self).__init__(*args, **kwargs)
        self.num_heads           = num_heads
        self.key_dim             = key_dim
        self.attn_height         = attn_height
        self.qk_scale            = qk_scale
        self.qv_bias             = qv_bias
        self.qkv_bias            = qkv_bias
        self.return_weight       = return_weight
        self.return_bias         = return_bias
        self.pos_emb             = pos_emb
        self.rotate_pos_emb      = rotate_pos_emb
        self.text_max_block_size = text_max_block_size
        self.attn_dropout        = attn_dropout

    def build(self, input_shape):
        self.bs, self.bb, self.cc = input_shape
        self.key_dim = self.key_dim if self.key_dim > 0 else self.cc // self.num_heads
        embed_dim = int(self.num_heads * self.key_dim)
        is_text_inputs = self.text_max_block_size > 0
                
        self.qkv_bias, self.qv_bias = (True, False) if self.qkv_bias else (False, self.qv_bias)
        self.qkv_project = Dense(embed_dim * 3, use_bias=self.qkv_bias)
        
        if self.qv_bias:
            self.query_bias = BiasLayer()
            self.value_bias = BiasLayer()

        if self.rotate_pos_emb and is_text_inputs:
            self.rope_layer = PositionalEncodingFourierRot1D(max_block_size=self.text_max_block_size)
        elif self.rotate_pos_emb:
            self.rope_layer = PositionalEncodingFourierRot(num_heads=self.num_heads,
                                                           attn_height=self.attn_height, 
                                                           cls_token=True)
        else:
            self.rope_layer = None

        if is_text_inputs:
            self.pos_emb_layer = CausalMask(block_size=self.text_max_block_size)
        elif self.pos_emb:
            self.pos_emb_layer = MultiHeadRelativePositionalEmbedding(attn_height=self.attn_height)
        else:
            self.pos_emb_layer = None

        if self.attn_dropout > 0:
            self.drop_layer = Dropout(self.attn_dropout)

        if self.return_weight:
            self.project_weight = Dense(self.cc, use_bias=self.return_bias)
            
    def scaled_dot_product_attention(self, query, key, value):
        scale_ratio = self.qk_scale if self.qk_scale > 0 else (1.0 / (float(query.shape[-1]) ** 0.5))
        attention_scores = query @ key

        if scale_ratio != 1:
            attention_scores = attention_scores * scale_ratio

        if self.pos_emb_layer is not None:
            attention_scores = self.pos_emb_layer(attention_scores)

        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        
        if self.attn_dropout > 0:
            attention_scores = self.drop_layer(attention_scores)

        attention_output = attention_scores @ value
        output = tf.transpose(attention_output, [0, 2, 1, 3])
        output = tf.reshape(output, [-1, self.bb, self.cc])

        if self.return_weight:
            output = self.project_weight(output)
            
        return output
    
    def call(self, inputs, training=False):
        qkv = self.qkv_project(inputs, training=training)
        query, key, value = tf.split(qkv, 3, axis=-1)

        if self.qv_bias:
            query = self.query_bias(query)
            value = self.value_bias(value)

        if isinstance(self.rope_layer, PositionalEncodingFourierRot1D):
            query = self.rope_layer(tf.reshape(query, [-1, self.num_heads, self.key_dim // 2, 2]))
            key = rope(tf.reshape(key, [-1, self.num_heads, self.key_dim // 2, 2]))
        elif isinstance(self.rope_layer, PositionalEncodingFourierRot):
            query = self.rope_layer(query)
            key = self.rope_layer(key)

        query = tf.reshape(query, [-1, self.bb, self.num_heads, self.key_dim])
        query = tf.transpose(query, [0, 2, 1, 3])

        key = tf.reshape(key, [-1, self.bb, self.num_heads, self.key_dim])
        key = tf.transpose(key, [0, 2, 3, 1])

        value = tf.reshape(value, [-1, self.bb, self.num_heads, self.key_dim])
        value = tf.transpose(value, [0, 2, 1, 3])
        return self.scaled_dot_product_attention(query, key, value)
