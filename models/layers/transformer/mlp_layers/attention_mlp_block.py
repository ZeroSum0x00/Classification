import tensorflow as tf
from tensorflow.keras.layers import add

from .mlp_block import MLPBlock
from models.layers import (
    get_normalizer_from_name,
    ChannelAffine, DropPathV1, DropPathV2,
)
from utils.model_processing import check_regularizer




class AttentionMLPBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        attention_layer,
        mlp_ratio=4,
        layer_scale=0.1,
        use_gated=False,
        activation="gelu",
        normalizer="layer-norm",
        use_mlp_norm=False,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_path_rate=0.0,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attention_layer = attention_layer
        self.mlp_ratio = mlp_ratio
        self.layer_scale = layer_scale
        self.use_gated = use_gated
        self.activation = activation
        self.normalizer = normalizer
        self.mlp_normalizer = normalizer if use_mlp_norm else None
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.mlp_block = MLPBlock(
            mlp_dim=int(input_shape[-1] * self.mlp_ratio),
            use_gated=self.use_gated,
            activation=self.activation,
            normalizer=self.mlp_normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            drop_rate=self.drop_rate,
        )
        
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        
        if self.layer_scale > 0:
            self.affine1 = ChannelAffine(use_bias=False, weight_init_value=self.layer_scale)
            self.affine2 = ChannelAffine(use_bias=False, weight_init_value=self.layer_scale)
            
        self.drop1 = DropPathV1(drop_prob=self.drop_path_rate)
        self.drop2 = DropPathV1(drop_prob=self.drop_path_rate)

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "attention_layer": self.attention_layer,
            "mlp_ratio": self.mlp_ratio,
            "layer_scale": self.layer_scale,
            "use_gated": self.use_gated,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "use_mlp_norm": self.use_mlp_norm,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "drop_path_rate": self.drop_path_rate,
            "drop_rate": self.drop_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        