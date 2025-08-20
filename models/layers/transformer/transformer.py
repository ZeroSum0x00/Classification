import tensorflow as tf
from tensorflow.keras.layers import Dropout

from models.layers import get_activation_from_name, get_normalizer_from_name



class TransformerEncoderBlock(tf.keras.layers.Layer):
    "Link: https://arxiv.org/pdf/1706.03762.pdf"

    def __init__(
        self,
        attention_block=None,
        mlp_block=None,
        activation=None,
        normalizer="layer-norm",
        norm_eps=1e-6,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attention_block = attention_block
        self.mlp_block = mlp_block
        self.activation = activation
        self.normalizer = normalizer
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activ_layer = get_activation_from_name(self.activation)
        self.dropout_layer1 = Dropout(rate=self.drop_rate)
        self.dropout_layer2 = Dropout(rate=self.drop_rate)

    def call(self, inputs, training=False, return_weight=False):
        x = self.norm_layer1(inputs, training=training)
        x, weights = self.attention_block(x, training=training, return_weight=return_weight)
        x = self.dropout_layer1(x, training=training)
        x = x + inputs
        
        y = self.norm_layer2(x, training=training)
        y = self.mlp_block(y, training=training)
        y = self.dropout_layer2(y, training=training)
        
        o = x + y
        o = self.activ_layer(o, training=training)
        return o, weights

    def get_config(self):
        config = super().get_config()
        config.update({
                "attention_block": self.attention_block,
                "mlp_block": self.mlp_block,
                "activation": self.activation,
                "normalizer": self.normalizer,
                "norm_eps": self.norm_eps,
                "drop_rate": self.drop_rate
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
