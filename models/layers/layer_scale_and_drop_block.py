import tensorflow as tf

from .stochastic_depth import DropPathV1, DropPathV2
from .channel_affine import ChannelAffine



class LayerScaleAndDropBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        layer_scale=0,
        residual_scale=0,
        drop_rate=0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layer_scale = layer_scale
        self.residual_scale = residual_scale
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.short_affine = ChannelAffine(use_bias=False, weight_init_value=self.residual_scale)
        self.deep_affine = ChannelAffine(use_bias=False, weight_init_value=self.layer_scale)
        self.drop_block = DropPathV1(drop_prob=self.drop_rate)

    def call(self, inputs, training=False):
        short, deep = inputs

        if self.residual_scale > 0:
            short = self.short_affine(short, training=training)

        if self.layer_scale > 0:
            deep = self.deep_affine(deep, training=training)

        deep = self.drop_block(deep, training=training)
        x = short + deep
        return x