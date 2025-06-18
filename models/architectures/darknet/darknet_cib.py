"""
    DarknetCIB: YOLOv10 Backbone with SCDown, C2f, and C2fCIB Blocks

    Overview:
        DarknetCIB is the backbone architecture used in YOLOv10, combining
        lightweight design with high feature representation efficiency. It
        introduces key components such as:
            - SCDown: Efficient downsampling module using shortcut connections
            - C2f: Light Cross-Stage Partial module (a variant of CSP with fewer layers)
            - C2fCIB: Channel Interaction Block for better multi-branch feature fusion

    Key Components:
        • SCDown:
            - A downsampling block that uses shortcut (residual) connections
              and grouped convolutions to preserve spatial information.
            - Typically used at the beginning of a new stage.

        • C2f (Cross-Stage Partial - fast/lightweight):
            - A simplified CSP block that splits the input, processes part
              through lightweight layers (e.g., bottlenecks), and concatenates.
            - Used to save parameters while preserving representational power.

        • C2fCIB (C2f with Channel Interaction Block):
            - Combines the efficiency of C2f with attention-like operations
              to model channel-level interactions across branches.
            - Enhances feature fusion and discrimination at low computational cost.
    
    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 640, 640, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=2) | (None, 320, 320, C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 160, 160, 2C)    |
        |                        | C2f (2x)                    | (None, 160, 160, 2C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | ConvolutionBlock (3x3, s=2) | (None, 80, 80, 4C)      |
        |                        | C2f (4x)                    | (None, 80, 80, 4C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | SCDown (3x3, s=2)           | (None, 40, 40, 8C)      |
        |                        | C2f (4x)                    | (None, 40, 40, 8C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | SCDown (3x3, s=2)           | (None, 20, 20, 16C*S)   |
        |                        | C2fCIB (2x)                 | (None, 20, 20, 16C*S)   |
        |                        | SPPF                        | (None, 20, 20, 16C*S)   |
        |                        | PSA                         | (None, 20, 20, 16C*S)   |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 16C*S)           |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------

    Model Parameter Comparison:
         ----------------------------------------------------------------------
        |      Model Name       |    Un-deploy params    |    Deploy params    |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB nano    |         1,464,536      |      1,464,536      |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB small   |         4,422,600      |      4,395,464      |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB medium  |         9,132,952      |      9,071,896      |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB base    |        11,778,728      |     11,724,456      |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB large   |        15,580,840      |     15,499,432      |
        |-----------------------+------------------------+---------------------|
        |    DarkNetCIB xlarge  |        15,831,640      |     15,526,360      |
         ----------------------------------------------------------------------

    References:
        - Paper: "YOLOv10: Real-Time End-to-End Object Detection"
          https://arxiv.org/pdf/2405.14458

        - Original implementation:
          https://github.com/THU-MIG/yolov10
          
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    concatenate, add
)

from .darknet19 import ConvolutionBlock
from .darknet_c3 import SPP, SPPF
from .darknet_c2 import C2f, LightConvolutionBlock

from models.layers import get_activation_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, create_model_backbone,
    create_layer_instance, validate_conv_arg, check_regularizer,
)



class SCDown(LightConvolutionBlock):

    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        kernel_size = validate_conv_arg(kernel_size)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.strides = validate_conv_arg(strides)

    def build(self, input_shape):
        super().build(input_shape)
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            groups=self.filters,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "strides": self.strides,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepVGGDW(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(7, 7),
            strides=(1, 1),
            groups=self.filters,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.filters,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        out = x1 + x2
        out = self.activ(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
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

        
class CIB(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        expansion=1.0,
        shortcut=True,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.expansion = expansion
        self.shortcut = shortcut  
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.deploy = deploy     
        
    def build(self, input_shape):
        self.c = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        if self.deploy:
            middle = ConvolutionBlock(
                filters=2 * hidden_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                groups=2 * hidden_dim,
                activation=self.normalizer,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
        else:
            middle = RepVGGDW(
                filters=2 * hidden_dim,
                activation=self.normalizer,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            
        self.block = Sequential([
            ConvolutionBlock(
                filters=self.c,
                kernel_size=(3, 3),
                strides=(1, 1),
                groups=self.c,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ),
            ConvolutionBlock(
                filters=2 * hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ),
            middle,
            ConvolutionBlock(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ),
            ConvolutionBlock(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                groups=self.filters,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ),
        ])

    def call(self, inputs, training=False):
        x = self.block(inputs, training=training)
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
            "shortcut": self.shortcut,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class C2fCIB(C2f):

    def __init__(
        self,
        filters,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.blocks = [
            CIB(
                filters=hidden_dim,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class SimpleAttention(tf.keras.layers.Layer):

    def __init__(
        self,
        dim,
        num_heads=8,
        attn_ratio=0.5,
        activation=None,
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        self.h = dim + nh_kd * 2

    def build(self, input_shape):
        self.qkv = ConvolutionBlock(
            filters=self.h,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.proj = ConvolutionBlock(
            filters=self.dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pe = ConvolutionBlock(
            filters=self.dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.dim,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
    def call(self, inputs, training=False):
        _, H, W, C = inputs.shape
        N = H * W

        x = self.qkv(inputs, training=training)
        x = tf.reshape(x, shape=[-1, N, self.num_heads, self.key_dim*2 + self.head_dim])
        q, k, v = tf.split(x, num_or_size_splits=[self.key_dim, self.key_dim, self.head_dim], axis=-1)
        attn = tf.transpose(q, perm=[0, 2, 1, 3]) @ tf.transpose(k, perm=[0, 2, 3, 1])
        attn *= self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.transpose(v, perm=[0, 2, 3, 1]) @ tf.transpose(attn, perm=[0, 1, 2, 3])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, H, W, C])
        x += self.pe(tf.reshape(v, shape=[-1, H, W, C]))
        x = self.proj(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "attn_ratio": self.attn_ratio,
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

        
class PSA(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        self.c = input_shape[-1]
        hidden_dim = int(self.c * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim * 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.c,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.attn = SimpleAttention(
            dim=hidden_dim,
            num_heads=hidden_dim // 64,
            attn_ratio=0.5,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.ffn = Sequential([
            ConvolutionBlock(
                filters=hidden_dim * 2,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ),
            ConvolutionBlock(
                filters=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=None,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        a, b = tf.split(x, num_or_size_splits=2, axis=-1)

        b += self.attn(b, training=training)
        b += self.ffn(b, training=training)        

        x = concatenate([a, b], axis=-1)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
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

        
def DarkNetCIB(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF", "PSA"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF, PSA].")

    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
        "deploy": deploy,
    }
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=640,
        min_size=32,
        weights=weights
    )

    if isinstance(feature_extractor, (tuple, list)):
       extractor_block1, extractor_block2 = feature_extractor
    else:
        extractor_block1 = extractor_block2 = feature_extractor


    if isinstance(fusion_layer, (list, tuple)):
        fusion_block1, fusion_block2 = fusion_layer
    else:
        fusion_block1 = fusion_block2 = fusion_layer

    if pyramid_pooling and not isinstance(pyramid_pooling, (list, tuple)):
        pyramid_pooling = [pyramid_pooling]

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks))]

    x = inputs
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            extractor_block1,
            filters=filters[0],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"
        
        f = filters[i + 1]

        if is_last_stage:
            f = int(f * final_channel_scale)
            final_filters = f
            
        if filters[0] < 80:
            _fusion_block = fusion_block1 if i < 3 else fusion_block2
        else:
            _fusion_block = fusion_block1 if i < 2 else fusion_block2

        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i < 2 else extractor_block2,
                filters=f,
                kernel_size=(3, 3),
                strides=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)

        if num_block > 1:
            x = create_layer_instance(
                _fusion_block,
                filters=f,
                iters=num_block - 1,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block2"
            )(x)

    block_name_prefix = f"stage{len(num_blocks) - 1}"
    
    if final_filters is None:
        final_filters = int(filters[-1] * final_channel_scale)
        
    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{p + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block3")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "DarkNet-CIB"
    if filters == [16, 32, 64, 128, 256] and num_blocks == [1, 2, 3, 3, 2]:
        model_name += "-nano"
    elif filters == [32, 64, 128, 256, 512] and num_blocks == [1, 2, 3, 3, 2]:
        model_name += "-small"
    elif filters == [48, 96, 192, 384, 768] and num_blocks == [1, 3, 5, 5, 3]:
        model_name += "-medium"
    elif filters == [64, 128, 256, 512, 1024] and num_blocks == [1, 3, 5, 5, 3]:
        model_name += "-base"
    elif filters == [64, 128, 256, 512, 1024] and num_blocks == [1, 4, 7, 7, 4]:
        model_name += "-large"
    elif filters == [80, 160, 320, 640, 1280] and num_blocks == [1, 4, 7, 7, 4]:
        model_name += "-xlarge"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetCIB_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block{j}" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]
    
    return create_model_backbone(
        model_fn=DarkNetCIB,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )

    
def DarkNetCIB_nano(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=C2f,
        pyramid_pooling=[SPPF, PSA],
        filters=16,
        num_blocks=[1, 2, 3, 3, 2],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model

def DarkNetCIB_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv10 version nano
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10n.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_nano,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )


def DarkNetCIB_small(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=[C2f, C2fCIB],
        pyramid_pooling=[SPPF, PSA],
        filters=32,
        num_blocks=[1, 2, 3, 3, 2],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def DarkNetCIB_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv10 version small
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10s.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_small,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )


def DarkNetCIB_medium(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=[C2f, C2fCIB],
        pyramid_pooling=[SPPF, PSA],
        filters=48,
        num_blocks=[1, 3, 5, 5, 3],
        channel_scale=2,
        final_channel_scale=0.75,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def DarkNetCIB_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv10 version medium
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10m.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_medium,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )


def DarkNetCIB_base(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=[C2f, C2fCIB],
        pyramid_pooling=[SPPF, PSA],
        filters=64,
        num_blocks=[1, 3, 5, 5, 3],
        channel_scale=2,
        final_channel_scale=0.5,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def DarkNetCIB_base_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv10 version base
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10b.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )


def DarkNetCIB_large(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=[C2f, C2fCIB],
        pyramid_pooling=[SPPF, PSA],
        filters=64,
        num_blocks=[1, 4, 7, 7, 4],
        channel_scale=2,
        final_channel_scale=0.5,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def DarkNetCIB_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv10 version large
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10l.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )


def DarkNetCIB_xlarge(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:

    model = DarkNetCIB(
        feature_extractor=[ConvolutionBlock, SCDown],
        fusion_layer=[C2f, C2fCIB],
        pyramid_pooling=[SPPF, PSA],
        filters=80,
        num_blocks=[1, 4, 7, 7, 4],
        channel_scale=2,
        final_channel_scale=0.5,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def DarkNetCIB_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[],
    deploy=False
) -> Model:
    
    """
        - Used in YOLOv10 version xlarge
        - In YOLOv10, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/THU-MIG/yolov10/blob/main/ultralytics/cfg/models/v10/yolov10x.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetCIB_xlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy
    )
