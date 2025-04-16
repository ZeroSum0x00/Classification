"""
  # Description:
    - The following table comparing the params of the CSP-DarkNet 53 (YOLOv4 backbone) in Tensorflow on 
    image size 416 x 416 x 3:

       ------------------------------------------
      |       Model Name      |     Params       |
      |------------------------------------------|
      |     CSP-DarkNet53     |    27,677,512    |
       ------------------------------------------

  # Reference:
    - Source: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
)

from .darknet53 import ConvolutionBlock, ResidualBlock
from models.layers import get_activation_from_name, LinearLayer
from utils.model_processing import process_model_input, create_layer_instance



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
        self.regularizer_decay = regularizer_decay
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

        
def CSPDarkNet53(
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
    pooling=None,
    activation="mish",
    normalizer="batch-norm",
    final_activation="softmax",
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

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock", "CSPDarkNetBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock, CSPDarkNetBlock].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

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
        default_size=640,
        min_size=32,
        weights=weights,
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
            name=f"stem.block{i + 1}",
        )(x)

    for i in range(len(num_blocks) - 1):
        f1 = filters[i + 1]
        f2 = [filters[i], filters[i + 1]] if i == 0 else [filters[i], filters[i]]
        
        x = create_layer_instance(
            extractor_block1 if i == 0 else extractor_block2,
            filters=int(f1 * final_channel_scale) if i == len(num_blocks) - 2 else f1,
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1",
        )(x)
        
        x = create_layer_instance(
            fusion_block1 if i == 0 else fusion_block2,
            filters=[int(f * final_channel_scale) for f in f2] if i == len(num_blocks) - 2 else f2,
            iters=num_blocks[i + 1],
            **layer_constant_dict,
            name=f"stage{i + 1}.block2",
        )(x)

    if pyramid_pooling:
        for j, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters[-1] * final_channel_scale),
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + 3}",
            )(x)
    else:
        x = LinearLayer(name=f"stage{i + 1}.block3")(x)
        
    if include_head:
        x = GlobalAveragePooling2D(name="global_avgpool")(x)
        x = Dropout(rate=drop_rate)(x)
        x = Dense(1 if num_classes == 2 else num_classes, name="predictions")(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 2, 8, 8, 4]:
        model = Model(inputs=inputs, outputs=x, name="CSPDarkNet-53-Base")
    else:
        model = Model(inputs=inputs, outputs=x, name="CSPDarkNet-53")
        
    return model


def CSPDarkNet53_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="mish",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    model = CSPDarkNet53(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [f"stage{i + 1}.block2" for i, j in enumerate(num_blocks[1:-1])]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def CSPDarkNet53_base(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="leaky-relu",
    normalizer="batch-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = CSPDarkNet53(
        feature_extractor=ConvolutionBlock,
        fusion_layer=CSPDarkNetBlock,
        pyramid_pooling=None,
        filters=32,
        num_blocks=[1, 1, 2, 8, 8, 4],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def CSPDarkNet53_base_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    """
        - Used in YOLOv4 base
        - In YOLOv4, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/yolov3/yolov4.py
    """
    
    model = CSPDarkNet53_base(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")