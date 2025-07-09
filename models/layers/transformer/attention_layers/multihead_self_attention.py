import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

from utils.model_processing import check_regularizer



class MultiHeadSelfAttention(tf.keras.layers.Layer):
    "Link: https://arxiv.org/pdf/1706.03762.pdf"
    
    def __init__(
        self,
        num_heads,
        num_embeds=-1,
        q_bias=True,
        kv_bias=False,
        use_causal_mask=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        drop_rate=0.,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.num_embeds = num_embeds
        self.q_bias = q_bias
        self.kv_bias = kv_bias
        self.use_causal_mask = use_causal_mask
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.drop_rate = drop_rate
        
    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):  
            if all(isinstance(shape, (list, tuple)) for shape in input_shape):
                query_shape = input_shape[0]
            else:  
                query_shape = input_shape
        elif isinstance(input_shape, tf.TensorShape):  
            query_shape = input_shape.as_list()
        else:  
            raise ValueError(f"Invalid input_shape type: {type(input_shape)}. Expected list, tuple, or TensorShape.")
        
        if not isinstance(query_shape, (list, tuple)) or len(query_shape) < 2:
            raise ValueError(f"Invalid query_shape: {query_shape}. Expected at least 2D tensor.")
        
        hidden_size = self.num_embeds if self.num_embeds != -1 else query_shape[-1]

        if hidden_size % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {self.num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // self.num_heads

        if isinstance(input_shape, (list, tuple)) and all(isinstance(shape, (list, tuple)) for shape in input_shape):
            if len(input_shape) == 2:
                self.query_project = Dense(
                    units=hidden_size,
                    use_bias=self.q_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
                self.keyvalue_project = Dense(
                    units=hidden_size * 2,
                    use_bias=self.kv_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
            elif len(input_shape) == 3:
                self.query_project = Dense(
                    units=hidden_size,
                    use_bias=self.q_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
                self.key_project = Dense(
                    units=hidden_size,
                    use_bias=self.kv_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
                self.value_project = Dense(
                    units=hidden_size,
                    use_bias=self.kv_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
            else:
                self.qkv_projection = Dense(
                    units=hidden_size * 3,
                    use_bias=self.q_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                )
                
        else:
            self.qkv_projection = Dense(
                units=hidden_size * 3,
                use_bias=self.q_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )


        self.combine_heads = Dense(
            units=hidden_size,
            use_bias=self.q_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.final_dropout = Dropout(rate=self.drop_rate)

        if self.use_causal_mask:
            block_size = query_shape[-2]
            causal_mask = np.tril(np.ones((block_size, block_size)))
            causal_mask = np.reshape(causal_mask, [1, block_size, block_size, 1])
            causal_mask = tf.convert_to_tensor(causal_mask, dtype=tf.float32)
            self.causal_mask = self.add_weight(
                shape=causal_mask.shape,
                initializer=causal_mask,
                trainable=True,
                name="causal_mask"
            )

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        attn = score / tf.math.sqrt(dim_key)

        if self.use_causal_mask:
            attn = tf.transpose(attn, [0, 2, 3, 1])
            mask = tf.where(tf.equal(self.causal_mask, 0), tf.zeros_like(self.causal_mask), self.causal_mask)
            attn = attn * mask
            attn = tf.transpose(attn, [0, 3, 1, 2])

        if attn_mask is not None:
            if attn_mask.dtype == tf.bool:
                attn = tf.where(attn_mask, -1e30, attn)
            else:
                attn += attn_mask
                
        weights = tf.nn.softmax(attn, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, attn_mask=None, training=False, return_weight=False):
        if hasattr(self, "key_project") and hasattr(self, "value_project"):
            query, key, value = inputs
            query = self.query_project(query, training=training)
            key   = self.key_project(key, training=training)
            value = self.value_project(value, training=training)
        elif hasattr(self, "keyvalue_project"):
            query, key_value = inputs
            query = self.query_project(query, training=training)
            kv    = self.keyvalue_project(key_value, training=training)
            key, value = tf.split(kv, 2, axis=-1)
        else:
            qkv = self.qkv_projection(inputs, training=training)
            query, key, value = tf.split(qkv, 3, axis=-1)

        batch_size = tf.shape(query)[0]
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        if attn_mask is not None:
            if tf.rank(attn_mask) == 2:
                attn_mask = tf.expand_dims(attn_mask, axis=0)

        attention, weights = self.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention, training=training)
        output = self.final_dropout(output, training=training)
        if return_weight:
            return output, weights
        else:
            return output, tf.constant(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_embeds": self.num_embeds,
            "q_bias": self.q_bias,
            "kv_bias": self.kv_bias,
            "use_causal_mask": self.use_causal_mask,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "drop_rate": self.drop_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        