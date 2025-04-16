"""
  # Description:
    - The following table comparing the params of the DarkNet 53 (YOLOv3 backbone) in Tensorflow on 
    image size 416 x 416 x 3:

       -----------------------------------------
      |      Model Name      |     Params       |
      |-----------------------------------------|
      |      DarkNet53       |    41,645,640    |
       -----------------------------------------

  # Reference:
    - Source: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Dense, Dropout,
    GlobalMaxPooling2D, GlobalAveragePooling2D, add
)
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import process_model_input, create_layer_instance


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
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
        if self.strides[0] > 1 or self.strides[1] > 1:
            self.padding = "valid"
        else:
            self.padding = "same"

    def build(self, input_shape):
        if self.padding == "valid":
            self.padding_layer = ZeroPadding2D(((1, 0), (1, 0)))
        
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
            kernel_regularizer=l2(self.regularizer_decay),
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
        self.regularizer_decay = regularizer_decay
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


def DarkNet53(
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
    activation="leaky-relu",
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

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "Conv2D" "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, Conv2D, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock].")

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
        f2 = [filters[i], filters[i + 1]]
        
        
        x = create_layer_instance(
            extractor_block1 if i == 0 else extractor_block2,
            filters=int(f1 * final_channel_scale) if i == len(num_blocks) - 2 else f1,
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1",
        )(x)
    
        for j in range(num_blocks[i + 1]):
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=[int(f * final_channel_scale) for f in f2] if i == len(num_blocks) - 2 else f2,
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + 2}",
            )(x)

    if pyramid_pooling:
        for k, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters[-1] * final_channel_scale),
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + k + 3}",
            )(x)
    else:
        x = LinearLayer(name=f"stage{i + 1}.block{j + 3}")(x)
        
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

    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 1, 2, 8, 8, 4]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-53-Base")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-53")
        
    return model


def DarkNet53_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    model = DarkNet53(
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

    custom_layers = custom_layers or [f"stage{i + 1}.block{j + 1}" for i, j in enumerate(num_blocks[1:-1])]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNet53_base(
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
    
    model = DarkNet53(
        feature_extractor=ConvolutionBlock,
        fusion_layer=ResidualBlock,
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


def DarkNet53_base_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    """
        - Used in YOLOv3 base
        - In YOLOv3, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/yolov3/yolov3.py
    """
    
    model = DarkNet53_base(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block3",
        "stage3.block9",
        "stage4.block9",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")