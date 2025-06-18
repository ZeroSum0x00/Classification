"""
    Overview:
        DarkNet-19 is a lightweight convolutional neural network introduced 
        as the backbone for YOLOv2 (You Only Look Once v2). It was designed 
        for real-time object detection with a balance between accuracy and speed.

        Compared to earlier versions, DarkNet-19 improves performance using 
        batch normalization, higher input resolution, and a deeper architecture 
        with more 3x3 convolutions.

    Key Characteristics:
        - 19 convolutional layers + 5 max-pooling layers
        - All convolution filters are 3x3, except the final 1x1 layer
        - Batch normalization applied to all convolutional layers
        - Uses Leaky ReLU activation
        - Designed for fast inference with relatively low parameter count

    General Model Architecture:
         ----------------------------------------------------------------------------------------
        | Stage                  | Layer                            | Output Shape               |
        |------------------------+----------------------------------+----------------------------|
        | Input                  | input_layer                      | (None, 416, 416, 3)        |
        |------------------------+----------------------------------+----------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=1)      | (None, 416, 416, C)        |
        |------------------------+----------------------------------+----------------------------|
        | Stage 1                | MaxPool2 (2x2, s=2)              | (None, 208, 208, C)        |
        |                        | ConvolutionBlock (3x3, s=1, t=1) | (None, 208, 208, 2C)       |
        |------------------------+----------------------------------+----------------------------|
        | Stage 2                | MaxPool2 (2x2, s=2)              | (None, 104, 104, 2C)       |
        |                        | ConvolutionBlock (3x3, s=1, t=3) | (None, 104, 104, 4C)       |
        |------------------------+----------------------------------+----------------------------|
        | Stage 3                | MaxPool2 (2x2, s=2)              | (None, 52, 52, 4C)         |
        |                        | ConvolutionBlock (3x3, s=1, t=3) | (None, 52, 52, 8C)         |
        |------------------------+----------------------------------+----------------------------|
        | Stage 4                | MaxPool2 (2x2, s=2)              | (None, 26, 26, 16C)        |
        |                        | ConvolutionBlock (3x3, s=1, t=5) | (None, 26, 26, 16C)        |
        |------------------------+----------------------------------+----------------------------|
        | Stage 5                | MaxPool2 (2x2, s=2)              | (None, 13, 13, 32C*S)      |
        |                        | ConvolutionBlock (3x3, s=1, t=5) | (None, 13, 13, 32C*S)      |
        |                        | extra_block_convs                | (None, 13, 13, 32C*S)      |
        |                        | skip_connection                  | (None, 13, 13, 32C*S + 8C) |
        |                        | extra_block_convs                | (None, 13, 13, 32C*S)      |
        |                        | pyramid_poolings (*)             | (None, 13, 13, 32C*S)      |
        |------------------------+----------------------------------+----------------------------|
        | CLS Logics             | Conv2D (1x1, s=1) (Logics)       | (None, 13, 13, 1000)       |
        |                        | GlobalAveragePooling2D           | (None, 1000)               |
         ----------------------------------------------------------------------------------------
        (*) Note: While the original architecture does not include a Pyramid Pooling layer, 
        it can be optionally incorporated to enhance feature aggregation and create an extended variant of the model.


    Model Parameter Comparison:
         -----------------------------------------
        |      Model Name      |     Params       |
        |----------------------+------------------|
        |      DarkNet19       |    51,572,936    |
         -----------------------------------------

    References:
        - Paper: "YOLO9000: Better, Faster, Stronger"
          https://arxiv.org/abs/1612.08242

        - Official implementation:
          https://github.com/pjreddie/darknet

        - PyTorch variant:
          https://github.com/eriklindernoren/PyTorch-YOLOv3
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Dense, LeakyReLU, MaxPooling2D,
    Dropout, GlobalAveragePooling2D,
    concatenate,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    SpaceToDepthV1, LinearLayer,
)
from utils.model_processing import (
    process_model_input, create_model_backbone,
    create_layer_instance, check_regularizer, validate_conv_arg,
)


class ConvolutionBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(1, 1),
        groups=1,
        activation="leaky-relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = validate_conv_arg(kernel_size)
        self.strides = validate_conv_arg(strides)
        self.dilation_rate = validate_conv_arg(dilation_rate)
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        
        if self.strides[0] > 1 or self.strides[1] > 1:
            self.padding = "valid"
        else:
            self.padding = "same"

    def build(self, input_shape):
        if self.padding == "valid":
            self.padding_layer = ZeroPadding2D(padding=((1, 0), (1, 0)))
        
        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            use_bias=not self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.normalizer = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activation = get_activation_from_name(self.activation)

    def call(self, inputs, training=False):
        if hasattr(self, "padding_layer"):
            inputs = self.padding_layer(inputs)

        x = self.conv(inputs, training=training)
        if self.normalizer:
            x = self.normalizer(x, training=training)
            
        if self.activation:
            x = self.activation(x)
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
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

        
def DarkNet19(
    feature_extractor,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=(416, 416, 3),
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "Conv2D" "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, Conv2D, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
    }
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=416,
        min_size=32,
        weights=weights
    )

    if isinstance(feature_extractor, (tuple, list)):
       extractor_block1, extractor_block2 = feature_extractor
    else:
        extractor_block1 = extractor_block2 = feature_extractor

    if pyramid_pooling and not isinstance(pyramid_pooling, (list, tuple)):
        pyramid_pooling = [pyramid_pooling]
        
    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks))]

    x = inputs
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            extractor_block1,
            filters=filters[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    skip_connection = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"
        
        for b in range(num_block):
            if b == 0:
                x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"{block_name_prefix}.block1")(x)
            else:
                f = filters[i] if b % 2 == 0 else filters[i + 1]
                k = (1, 1) if b % 2 == 0 else (3, 3)

                if is_last_stage:
                    f = int(f * final_channel_scale)
                    final_filters = f

                x = create_layer_instance(
                    extractor_block1 if i == 0 else extractor_block2,
                    filters=f,
                    kernel_size=k,
                    strides=(1, 1),
                    **layer_constant_dict,
                    name=f"{block_name_prefix}.block{b + 1}"
                )(x)
                
        if i == len(num_blocks[1:]) - 2:
            skip_connection = x
            
    block_name_prefix = f"stage{len(num_blocks) - 1}"

    if final_filters is None:
        final_filters = int(filters[-1] * final_channel_scale)

    x = Sequential([
        create_layer_instance(
            extractor_block2,
            filters=final_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            **layer_constant_dict,
        ),
        create_layer_instance(
            extractor_block2,
            filters=final_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            **layer_constant_dict,
        )
    ], name=f"{block_name_prefix}.block{b + 2}")(x)

    if skip_connection is not None:
        skip_connection = Sequential([
            create_layer_instance(
                extractor_block2,
                filters=filters[1],
                kernel_size=(1, 1),
                strides=(1, 1),
                **layer_constant_dict,
            ),
            SpaceToDepthV1(block_size=2),
        ], name=f"{block_name_prefix}.skip_connection")(skip_connection)
    
        x = concatenate([skip_connection, x], name=f"{block_name_prefix}.concat")

    x = create_layer_instance(
        extractor_block2,
        filters=final_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name=f"{block_name_prefix}.block{b + 3}"
    )(x)

    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{b + p + 4}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block{b + 4}")(x)
        
    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Conv2D(
                filters=1 if num_classes == 2 else num_classes,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
            ),
            GlobalAveragePooling2D(),
        ], name="classifier_head")(x)

    model_name = "DarkNet-19"
    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 2, 4, 4, 6, 6]:
        model_name += "-base"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNet19_backbone(
    feature_extractor,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [f"stage{i + 1}.block{j}" for i, j in enumerate(num_blocks[1:-1])]

    return create_model_backbone(
        model_fn=DarkNet19,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNet19_base(
    inputs=[416, 416, 3],
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNet19(
        feature_extractor=ConvolutionBlock,
        pyramid_pooling=None,
        filters=32,
        num_blocks=[1, 2, 4, 4, 6, 6],
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
        drop_rate=drop_rate
    )
    return model


def DarkNet19_base_backbone(
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block4",
        "stage3.block4",
        "stage4.block6",
    ]

    return create_model_backbone(
        model_fn=DarkNet19_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    