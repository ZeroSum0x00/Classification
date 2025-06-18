"""
    EfficientNet: Scalable and Efficient Backbone with Compound Scaling
    
    Overview:
        EfficientNet is a family of convolutional neural networks that balance
        accuracy and efficiency using a principled compound scaling method.
        It uniformly scales network width, depth, and resolution with a fixed 
        set of scaling coefficients, achieving better accuracy and efficiency 
        than previous ConvNets at a lower computational cost.
    
        Core design choices include:
            - MBConv Block: Mobile inverted bottleneck convolution with squeeze-and-excitation
            - Compound Scaling: A principled way to scale all dimensions of the network
            - Depthwise Separable Convolution: Efficient spatial feature extraction
    
    Key Components:
        • MBConv (Mobile Inverted Bottleneck Convolution):
            - A depthwise separable convolution block with inverted residual structure.
            - Expands input channels → depthwise convolution → squeeze-and-excitation → project to output channels.
            - Residual connection is used when input and output shapes match.
            - Helps preserve accuracy while reducing parameters and FLOPs.
    
        • Squeeze-and-Excitation (SE) Module:
            - Channel attention mechanism used inside MBConv.
            - Applies global average pooling → 1x1 FC down → ReLU → 1x1 FC up → Sigmoid.
            - Output is used to reweight channels, enhancing informative features.
    
        • Compound Scaling:
            - Instead of arbitrarily scaling network dimensions, EfficientNet uses a 
              compound coefficient φ to scale depth (d), width (w), and input resolution (r):
              ```
              depth ∝ α^φ, width ∝ β^φ, resolution ∝ γ^φ, where α·β²·γ² ≈ 2
              ```
            - This method ensures balanced scaling for better performance.
    
        • Stage-wise Structure:
            - The network is divided into stages, each containing repeated MBConv blocks 
              with increasing depth and output channels.
            - Downsampling is performed via stride-2 convolutions at the start of a new stage.
    
        • EfficientNet Variants:
            - EfficientNet-B0: Base model
            - EfficientNet-B1 to B7: Scaled versions using compound scaling
            - EfficientNetV2: An improved variant with better training speed and accuracy

    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 224, 224, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ZeroPadding2D (1x1)         | (None, 225, 225, 3)     |
        |                        | ConvolutionBlock (3x3, s=2) | (None, 112, 112, 48)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | efficient_block (2x)        | (None, 112, 112, 24)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | efficient_block (4x)        | (None, 56, 56, 32)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | efficient_block (4x)        | (None, 28, 28, 56)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | efficient_block (6x)        | (None, 14, 14, 112)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 5                | efficient_block (6x)        | (None, 14, 14, 160)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 6                | efficient_block (8x)        | (None, 7, 7, 272)       |
        |------------------------+-----------------------------+-------------------------|
        | Stage 7                | efficient_block (2x)        | (None, 7, 7, 448)       |
        |                        | ConvolutionBlock (1x1, s=1) | (None, 7, 7, 1792)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 1792)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------

    Model Parameter Comparison:
         ------------------------------------------
        |     Model Name         |     Params      |
        |------------------------------------------|
        |     EfficientNet-B0    |     5,330,564   |
        |------------------------|-----------------|
        |     EfficientNet-B1    |     7,856,232   |
        |------------------------|-----------------|
        |     EfficientNet-B2    |     9,177,562   |
        |------------------------|-----------------|
        |     EfficientNet-B3    |    12,320,528   |
        |------------------------|-----------------|
        |     EfficientNet-B4    |    19,466,816   |
        |------------------------|-----------------|
        |     EfficientNet-B5    |    30,562,520   |
        |------------------------|-----------------|
        |     EfficientNet-B6    |    43,265,136   |
        |------------------------|-----------------|
        |     EfficientNet-B7    |    66,658,680   |
         ------------------------------------------
    
    References:
        - Paper: “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”  
          https://arxiv.org/abs/1905.11946

        - Official implementation:  
          https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
        
"""

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, ZeroPadding2D, DepthwiseConv2D,
    Reshape, Dropout, GlobalAveragePooling2D,
    multiply, add,
)

from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, correct_pad,
    check_regularizer, validate_conv_arg,
    create_model_backbone,
)



DEFAULT_BLOCKS_ARGS = [
    {"filters_in": 32, "filters_out": 16, "kernel_size": 3, "strides": 1,
     "expand_ratio": 1, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 1},
    {"filters_in": 16, "filters_out": 24, "kernel_size": 3, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 2},
    {"filters_in": 24, "filters_out": 40, "kernel_size": 5, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 2},
    {"filters_in": 40, "filters_out": 80, "kernel_size": 3, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 3},
    {"filters_in": 80, "filters_out": 112, "kernel_size": 5, "strides": 1,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 3},
    {"filters_in": 112, "filters_out": 192, "kernel_size": 5, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 4},
    {"filters_in": 192, "filters_out": 320, "kernel_size": 3, "strides": 1,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 1},
]


def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def efficient_block(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    expand_ratio=1,
    squeeze_ratio=0.,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    residual_connection=True,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"efficient_block_{K.get_uid('efficient_block')}"

    f1, f2 = filters
    f = f1 * expand_ratio
    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)
    
    if expand_ratio != 1:
        x = Sequential([
                Conv2D(
                filters=f,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
        ], name=f"{name}.expand")(inputs)
    else:
        x = inputs

    x = Sequential([
        ZeroPadding2D(padding=correct_pad(x, kernel_size)) if tuple(strides) == (2, 2) else LinearLayer(),
        DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding="valid" if tuple(strides) == (2, 2) else "same",
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.dwconv")(x)

    if 0 < squeeze_ratio <= 1:
        filters_squeeze = max(1, int(f1 * squeeze_ratio))

        squeeze = Sequential([
            GlobalAveragePooling2D(),
            Reshape(target_shape=[1, 1, f]),
            Conv2D(
                filters=filters_squeeze,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            Conv2D(
                filters=f,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                activation="sigmoid",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
        ], name=f"{name}.squeeze")(x)
        
        x = multiply([x, squeeze], name=f"{name}.excite")
    
    x = Sequential([
        Conv2D(
            filters=f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.project")(x)
        

    if (residual_connection is True) and (strides == 1) and (f1 == f2):
        if drop_rate > 0:
            x = Dropout(rate=drop_rate, noise_shape=(None, 1, 1, 1), name=f"{name}.drop")(x)
        else:
            x = LinearLayer(name=f"{name}.drop")(x)
            
        x = add([x, inputs], name=f"{name}.add")
    else:
        x = LinearLayer(name=f"{name}.add")(x)
    return x


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    depth_divisor=8,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')
        
    regularizer_decay = check_regularizer(regularizer_decay)

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    x = Sequential([
        ZeroPadding2D(padding=correct_pad(inputs, 3)),
        Conv2D(
            filters=round_filters(32, width_coefficient, depth_divisor),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)
    
    b = 0
    blocks = float(sum(args["repeats"] for args in blocks_args))
    for idx, args in enumerate(blocks_args):
        filters_in = round_filters(args["filters_in"], width_coefficient, depth_divisor)
        filters_out = round_filters(args["filters_out"], width_coefficient, depth_divisor)
        kernel_size = args["kernel_size"]
        strides = args["strides"]
        expand_ratio = args["expand_ratio"]
        squeeze_ratio = args["squeeze_ratio"]
        residual_connection = args["residual_connection"]
        repeats = args["repeats"]

        for i in range(round_repeats(repeats, depth_coefficient)):
            if i > 0:
                strides = 1
                filters_in = filters_out

            x = efficient_block(
                inputs=x,
                filters=[filters_in, filters_out],
                kernel_size=kernel_size,
                strides=strides,
                expand_ratio=expand_ratio,
                squeeze_ratio=squeeze_ratio,
                activation=activation,
                normalizer=normalizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                regularizer_decay=regularizer_decay,
                norm_eps=norm_eps,
                residual_connection=residual_connection,
                drop_rate=drop_connect_rate * b / blocks,
                name=f"stage{idx + 1}.block{i + 1}"
            )
            b += 1

    x = Sequential([
        Conv2D(
            round_filters(1280, width_coefficient, depth_divisor),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage{idx + 1}.block{i + 2}")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "EfficientNet"
    if (width_coefficient == 1.0) and (depth_coefficient == 1.0):
        model_name += "-B0"
    elif (width_coefficient == 1.0) and (depth_coefficient == 1.1):
        model_name += "-B1"
    elif (width_coefficient == 1.1) and (depth_coefficient == 1.2):
        model_name += "-B2"
    elif (width_coefficient == 1.2) and (depth_coefficient == 1.4):
        model_name += "-B3"
    elif (width_coefficient == 1.4) and (depth_coefficient == 1.8):
        model_name += "-B4"
    elif (width_coefficient == 1.6) and (depth_coefficient == 2.2):
        model_name += "-B5"
    elif (width_coefficient == 1.8) and (depth_coefficient == 2.6):
        model_name += "-B6"
    elif (width_coefficient == 2.0) and (depth_coefficient == 3.1):
        model_name += "-B7"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def EfficientNet_backbone(
    width_coefficient,
    depth_coefficient,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    depth_divisor=8,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    get_stages = [0, 1, 2, 4]
    get_numblocks = [round_repeats(r["repeats"], depth_coefficient) for i, r in enumerate(blocks_args) if i in get_stages]
    custom_layers = custom_layers or [f"stage{s + 1}.block{b}.add" for s, b in zip(get_stages, get_numblocks)]

    return create_model_backbone(
        model_fn=EfficientNet,
        custom_layers=custom_layers,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        blocks_args=blocks_args,
        depth_divisor=depth_divisor,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB0(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block1.add",
        "stage2.block2.add",
        "stage3.block2.add",
        "stage5.block3.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB0,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.2,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.add",
        "stage2.block3.add",
        "stage3.block3.add",
        "stage5.block4.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.3,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.add",
        "stage2.block3.add",
        "stage3.block3.add",
        "stage5.block4.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB3(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.3,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.add",
        "stage2.block3.add",
        "stage3.block3.add",
        "stage5.block5.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB3,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def EfficientNetB4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.4,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.4,
        depth_coefficient=1.8,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.add",
        "stage2.block4.add",
        "stage3.block4.add",
        "stage5.block6.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def EfficientNetB5(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.4,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.6,
        depth_coefficient=2.2,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB5_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.add",
        "stage2.block5.add",
        "stage3.block5.add",
        "stage5.block7.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB5,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB6(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.8,
        depth_coefficient=2.6,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB6_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.add",
        "stage2.block6.add",
        "stage3.block6.add",
        "stage5.block8.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB6,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def EfficientNetB7(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=2.0,
        depth_coefficient=3.1,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
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
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB7_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block4.add",
        "stage2.block7.add",
        "stage3.block7.add",
        "stage5.block10.add",
    ]

    return create_model_backbone(
        model_fn=EfficientNetB7,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
