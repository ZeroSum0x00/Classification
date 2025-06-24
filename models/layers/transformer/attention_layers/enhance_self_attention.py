import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from models.layers import BiasLayer
from ..causal_mask import CausalMask
from ..positional_encoding_fourier_rot import PositionalEncodingFourierRot1D, PositionalEncodingFourierRot
from ..multihead_relative_positional_embedding import MultiHeadRelativePositionalEmbedding
from utils.model_processing import check_regularizer



class EnhanceSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
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
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn_height = attn_height
        self.qk_scale = qk_scale
        self.qv_bias = qv_bias
        self.qkv_bias = qkv_bias
        self.return_weight = return_weight
        self.return_bias = return_bias
        self.pos_emb = pos_emb
        self.rotate_pos_emb = rotate_pos_emb
        self.text_max_block_size = text_max_block_size
        self.attn_dropout = attn_dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        
    def build(self, input_shape):
        self.bs, self.bb, self.cc = input_shape
        self.key_dim = self.key_dim if self.key_dim > 0 else self.cc // self.num_heads
        embed_dim = int(self.num_heads * self.key_dim)
        is_text_inputs = self.text_max_block_size > 0
                
        self.qkv_bias, self.qv_bias = (True, False) if self.qkv_bias else (False, self.qv_bias)
        self.qkv_project = Dense(
            units=embed_dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        if self.qv_bias:
            self.query_bias = BiasLayer()
            self.value_bias = BiasLayer()

        if self.rotate_pos_emb and is_text_inputs:
            self.rope_layer = PositionalEncodingFourierRot1D(max_block_size=self.text_max_block_size)
        elif self.rotate_pos_emb:
            self.rope_layer = PositionalEncodingFourierRot(
                num_heads=self.num_heads,
                attn_height=self.attn_height,
                cls_token=True,
            )
        else:
            self.rope_layer = None

        if is_text_inputs:
            self.pos_emb_layer = CausalMask(block_size=self.text_max_block_size)
        elif self.pos_emb:
            self.pos_emb_layer = MultiHeadRelativePositionalEmbedding(
                num_heads=-1,
                attn_height=self.attn_height,
                cls_token=True,
            )
        else:
            self.pos_emb_layer = None

        if self.attn_dropout > 0:
            self.drop_layer = Dropout(self.attn_dropout)

        if self.return_weight:
            self.project_weight = Dense(
                units=self.cc,
                use_bias=self.return_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )
            
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
            key = self.rope_layer(tf.reshape(key, [-1, self.num_heads, self.key_dim // 2, 2]))
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "attn_height": self.attn_height,
            "qk_scale": self.qk_scale,
            "qv_bias": self.qv_bias,
            "qkv_bias": self.qkv_bias,
            "return_weight": self.return_weight,
            "return_bias": self.return_bias,
            "pos_emb": self.pos_emb,
            "rotate_pos_emb": self.rotate_pos_emb,
            "text_max_block_size": self.text_max_block_size,
            "attn_dropout": self.attn_dropout,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    