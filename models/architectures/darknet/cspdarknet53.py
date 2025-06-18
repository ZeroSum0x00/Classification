"""
    Overview:
        CSPDarkNet-53 (Cross Stage Partial DarkNet-53) is an enhanced version of the 
        DarkNet-53 architecture used as a backbone in object detection models like YOLOv4. 
        It introduces **Cross Stage Partial (CSP) connections** to improve learning capability 
        while reducing computational cost and redundancy in feature maps.

        CSPNet divides the feature map of the base layer into two parts and then merges them 
        through a cross-stage hierarchy. This design improves gradient flow, reduces 
        computation, and enhances overall accuracy.

    Key Characteristics:
        - Based on DarkNet-53, a ResNet-style architecture with residual blocks
        - Incorporates CSP connections to improve efficiency and reduce redundancy
        - Improved accuracy and speed over traditional backbones like ResNet50/101
        - Commonly used in YOLOv4, YOLOv5, and other real-time object detection frameworks
        - Efficient on edge devices due to low computation complexity

    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 416, 416, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=1) | (None, 416, 416, C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 208, 208, 2C)    |
        |                        | CSPDarkNetBlock (1x)        | (None, 208, 208, 2C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | ConvolutionBlock (3x3, s=2) | (None, 104, 104, 4C)    |
        |                        | CSPDarkNetBlock (2x)        | (None, 104, 104, 4C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | ConvolutionBlock (3x3, s=2) | (None, 52, 52, 8C)      |
        |                        | CSPDarkNetBlock (8x)        | (None, 52, 52, 8C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | ConvolutionBlock (3x3, s=2) | (None, 26, 26, 16C)     |
        |                        | CSPDarkNetBlock (8x)        | (None, 26, 26, 16C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 5                | ConvolutionBlock (3x3, s=2) | (None, 13, 13, 32C*S)   |
        |                        | CSPDarkNetBlock (4x)        | (None, 13, 13, 32C*S)   |
        |                        | pyramid_poolings (*)        | (None, 13, 13, 32C*S)   |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 32C*S)           |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
        (*) Note: While the original architecture does not include a Pyramid Pooling layer, 
        it can be optionally incorporated to enhance feature aggregation and create an extended variant of the model.

    Model Parameter Comparison:
         ------------------------------------------
        |       Model Name      |     Params       |
        |-----------------------+------------------|
        |     CSPDarkNet-53     |    27,677,512    |
         ------------------------------------------

    References:
        - Paper: "CSPNet: A New Backbone that can Enhance Learning Capability of CNN"
          https://arxiv.org/abs/1911.11929

        - YOLOv4 paper: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
          https://arxiv.org/abs/2004.10934

        - PyTorch implementation:
          https://github.com/WongKinYiu/CrossStagePartialNetworks
          
        - TensorFlow/Keras repository:
          https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
"""


import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    concatenate,
)

from .darknet19 import ConvolutionBlock
from .darknet53 import ResidualBlock
from models.layers import get_activation_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, create_model_backbone,
    create_layer_instance, check_regularizer,
)



class CSPDarkNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        iters=1,
        activation="mish",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.iters = iters
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.shortcut = ConvolutionBlock(
            filters=self.filters[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv1 = ConvolutionBlock(
            filters=self.filters[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.middle = Sequential([
            ResidualBlock(
                filters=self.filters[:2],
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ) for _ in range(self.iters)
        ])
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv3 = ConvolutionBlock(
            filters=self.filters[0] * 2,
            kernel_size=(1, 1),
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
        x = self.middle(x, training=training)
        x = self.conv2(x, training=training)
        y = self.shortcut(inputs, training=training)

        merger = concatenate([x, y], axis=-1)
        merger = self.conv3(merger, training=training)
        return merger

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "iters": self.iters,
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

        
def CSPDarkNet53(
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
    activation="mish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
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

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock", "CSPDarkNetBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock, CSPDarkNetBlock].")

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
        f2 = [filters[i], filters[i + 1]] if i == 0 else [filters[i], filters[i]]
    
        if is_last_stage:
            f1 = int(f1 * final_channel_scale)
            f2 = [int(f * final_channel_scale) for f in f2]
            final_filters = f1
            
        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i == 0 else extractor_block2,
                filters=f1,
                kernel_size=(3, 3),
                strides=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)

        if num_block > 1:
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=f2,
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

    model_name = "CSPDarkNet-53"
    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 2, 8, 8, 4]:
        model_name += "-base"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def CSPDarkNet53_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="mish",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [f"stage{i + 1}.block2" for i, j in enumerate(num_blocks[1:-1])]

    return create_model_backbone(
        model_fn=CSPDarkNet53,
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


def CSPDarkNet53_base(
    inputs=[416, 416, 3],
    include_head=True,
    weights="imagenet",
    activation="mish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = CSPDarkNet53(
        feature_extractor=ConvolutionBlock,
        fusion_layer=CSPDarkNetBlock,
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


def CSPDarkNet53_base_backbone(
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="mish",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv4 base
        - In YOLOv4, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/yolov3/yolov4.py
    """

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    return create_model_backbone(
        model_fn=CSPDarkNet53_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    