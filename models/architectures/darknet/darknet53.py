"""
    Overview:
        DarkNet-53 is a convolutional neural network architecture introduced as 
        the backbone for YOLOv3, designed for object detection tasks. It is a deeper, 
        more accurate, and more efficient successor to DarkNet-19 used in YOLOv2.

        The architecture combines successive 3x3 and 1x1 convolutional layers with 
        residual connections, inspired by ResNet, to ease training and enable deeper networks.

    Key Characteristics:
        - 53 convolutional layers
        - Residual (shortcut) connections to improve gradient flow
        - Batch normalization after each convolution
        - Leaky ReLU activation
        - No fully connected layers
        - Designed for real-time object detection with high accuracy

    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 416, 416, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=1) | (None, 416, 416, C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 208, 208, 2C)    |
        |                        | ResidualBlock (1x)          | (None, 208, 208, 2C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | ConvolutionBlock (3x3, s=2) | (None, 104, 104, 4C)    |
        |                        | ResidualBlock (1x)          | (None, 104, 104, 4C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | ConvolutionBlock (3x3, s=2) | (None, 52, 52, 8C)      |
        |                        | ResidualBlock (1x)          | (None, 52, 52, 8C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | ConvolutionBlock (3x3, s=2) | (None, 26, 26, 16C)     |
        |                        | ResidualBlock (1x)          | (None, 26, 26, 16C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 5                | ConvolutionBlock (3x3, s=2) | (None, 13, 13, 32C*S)   |
        |                        | ResidualBlock (1x)          | (None, 13, 13, 32C*S)   |
        |                        | pyramid_poolings (*)        | (None, 13, 13, 32C*S)   |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 32C*S)           |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
        (*) Note: While the original architecture does not include a Pyramid Pooling layer, 
        it can be optionally incorporated to enhance feature aggregation and create an extended variant of the model.

    Model Parameter Comparison:
         -----------------------------------------
        |      Model Name      |     Params       |
        |----------------------+------------------|
        |      DarkNet53       |    41,645,640    |
         -----------------------------------------

    References:
        - Paper: "YOLOv3: An Incremental Improvement"
          https://arxiv.org/abs/1804.02767

        - Official implementation:
          https://github.com/pjreddie/darknet

        - PyTorch implementation:
          https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
"""


import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Dense,
    Dropout, GlobalAveragePooling2D,
    add,
)

from .darknet19 import ConvolutionBlock
from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, create_model_backbone,
    create_layer_instance, check_regularizer, validate_conv_arg,
)



class ResidualBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
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
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        if isinstance(self.filters, int):
            f0 = f1 = self.filters
        else:
            f0, f1 = self.filters
            
        self.conv1 = ConvolutionBlock(
            filters=f0,
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
            filters=f1,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return add([inputs, x])

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

    
def DarkNet53(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
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
        default_size=[416, 512, 608],
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
            strides=(1, 1),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        f1 = filters[i + 1]
        f2 = [filters[i], filters[i + 1]]

        if is_last_stage:
            f1 = int(f1 * final_channel_scale)
            f2 = [int(f * final_channel_scale) for f in f2]
            final_filters = f1
        
        for b in range(num_block):
            if b == 0:
                x = create_layer_instance(
                    extractor_block1 if i == 0 else extractor_block2,
                    filters=f1,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    **layer_constant_dict,
                    name=f"{block_name_prefix}.block1"
                )(x)
            else:
                x = create_layer_instance(
                    fusion_block1 if i == 0 else fusion_block2,
                    filters=f2,
                    **layer_constant_dict,
                    name=f"{block_name_prefix}.block{b + 2}"
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
                name=f"{block_name_prefix}.block{b + p + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block{b + 3}")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "DarkNet-53"
    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 2, 3, 9, 9, 5]:
        model_name += "-base"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNet53_backbone(
    feature_extractor,
    fusion_layer,
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

    custom_layers = custom_layers or [f"stage{i + 1}.block{j + 1}" for i, j in enumerate(num_blocks[1:-1])]

    return create_model_backbone(
        model_fn=DarkNet53,
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
        normalizer=normalizer
    )


def DarkNet53_base(
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
    
    model = DarkNet53(
        feature_extractor=ConvolutionBlock,
        fusion_layer=ResidualBlock,
        pyramid_pooling=None,
        filters=32,
        num_blocks=[1, 2, 3, 9, 9, 5],
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


def DarkNet53_base_backbone(
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv3 base
        - In YOLOv3, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/yolov3/yolov3.py
    """

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block3",
        "stage3.block9",
        "stage4.block9",
    ]

    return create_model_backbone(
        model_fn=DarkNet53_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    