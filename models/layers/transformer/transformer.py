import tensorflow as tf
from tensorflow.keras.layers import Dropout

from models.layers import get_activation_from_name, get_normalizer_from_name



class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        attn_block=None,
        ffn_block=None,
        activation=None,
        normalizer="layer-norm",
        norm_eps=1e-6,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attn_block = attn_block
        self.ffn_block = ffn_block
        self.activation = activation
        self.normalizer = normalizer
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer3 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activ_layer = get_activation_from_name(self.activation)
        self.dropout = Dropout(rate=self.drop_rate)

    def call(self, inputs, attn_mask=None, training=False, return_weight=False):
        hidden_state = self.norm_layer1(inputs, training=training)
        attn_out, weights = self.attn_block(
            hidden_state, attn_mask=attn_mask, training=training, return_weight=return_weight
        )
        hidden_state = hidden_state + attn_out
        
        hidden_state = self.norm_layer2(hidden_state, training=training)
        ffn_out = self.ffn_block(hidden_state, training=training)
        hidden_state = hidden_state + ffn_out
        
        hidden_state = self.norm_layer3(hidden_state, training=training)
        hidden_state = self.activ_layer(hidden_state, training=training)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "attn_block": self.attn_block,
            "ffn_block": self.ffn_block,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "norm_eps": self.norm_eps,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerDecoderBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        masked_attn_block=None,
        cross_attn_block=None,
        ffn_block=None,
        activation=None,
        normalizer="layer-norm",
        norm_eps=1e-6,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.masked_attn_block = masked_attn_block
        self.cross_attn_block = cross_attn_block
        self.ffn_block = ffn_block
        self.activation = activation
        self.normalizer = normalizer
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        if self.cross_attn_block is not None:
            self.norm_layer_cross = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
            
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer3 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activ_layer = Activation(self.activation)
        self.dropout = Dropout(self.drop_rate)

    def call(self, inputs, self_mask=None, cross_mask=None, training=False, return_weight=False):
        hidden_state, encoder_outputs = inputs
        hidden_state = self.norm_layer1(hidden_state, training=training)
        masked_attn_out, weights1 = self.masked_attn_block(
            hidden_state, attn_mask=self_mask, training=training, return_weight=return_weight
        )
        hidden_state = hidden_state + masked_attn_out
        
        if self.cross_attn_block is not None:
            hidden_state = self.norm_layer_cross(hidden_state, training=training)
            cross_attn_out, weights2 = self.cross_attn_block(
                [hidden_state, encoder_outputs], attn_mask=cross_mask, training=training, return_weight=return_weight
            )
            hidden_state = hidden_state + cross_attn_out
        else:
            weights2 = tf.constant(0.0)

        hidden_state = self.norm_layer2(hidden_state, training=training)
        ffn_out = self.ffn_block(hidden_state, training=training)        
        hidden_state = hidden_state + ffn_out
        
        hidden_state = self.norm_layer3(hidden_state, training=training)
        hidden_state = self.activ_layer(hidden_state, training=training)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state, weights1, weights2

    def get_config(self):
        config = super().get_config()
        config.update({
            "masked_attn_block": self.masked_attn_block,
            "cross_attn_block": self.cross_attn_block,
            "ffn_block": self.ffn_block,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "norm_eps": self.norm_eps,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
