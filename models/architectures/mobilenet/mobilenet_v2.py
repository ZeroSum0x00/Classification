"""
    MobileNetV2: Lightweight Backbone with Inverted Residuals and Linear Bottlenecks
    
    Overview:
        MobileNetV2 improves upon MobileNetV1 by introducing two key ideas to enhance
        efficiency and representational power:
            - **Inverted Residual Blocks** with linear bottlenecks
            - **Shortcuts (Residual connections)** for better gradient flow
        
        It is designed for high performance on mobile and edge devices, balancing
        accuracy and computational cost.
    
        Key innovations include:
            - Inverted Residual Structure: Expands → Depthwise → Project (compress)
            - Linear Bottleneck: Avoids non-linearity after compression to preserve features
            - Lightweight yet expressive thanks to depthwise separable convs and skip connections
    
    Key Components:
        • Inverted Residual Block:
            - Core unit of MobileNetV2 with the following structure:
            - **Expansion Layer**: Projects input to higher dimension (t× input channels)
            - **Depthwise Convolution**: Applies spatial filtering per channel
            - **Projection Layer**: Compresses back to output dimension (no ReLU)
            - **Residual Connection**: Added when input and output shapes match
    
        • Linear Bottleneck:
            - The final 1×1 conv (projection layer) has no activation (linear)
            - Prevents information loss when compressing back to low dimensions

        • Width Multiplier (α):
            - Controls the number of channels throughout the network
            - Allows model size to scale smoothly (α = 1.0, 0.75, 0.5, etc.)
    
        • Stride & Downsampling:
            - Downsampling is performed in selected bottleneck blocks via stride = 2
            - Ensures efficient spatial reduction

    General Model Architecture:
         ------------------------------------------------------------------------------------
        | Stage                  | Layer                           | Output Shape            |
        |------------------------+---------------------------------+-------------------------|
        | Input                  | input_layer                     | (None, 224, 224, 3)     |
        |------------------------+---------------------------------+-------------------------|
        | Stem                   | ZeroPadding2D (1x1)             | (None, 225, 225, 3)     |
        |                        | ConvolutionBlock (3x3, s=2)     | (None, 112, 112, 32)    |
        |------------------------+---------------------------------+-------------------------|
        | Stage 1                | inverted_residual_block         | (None, 112, 112, 16)    |
        |                        | inverted_residual_block (s=2)   | (None, 56, 56, 24)      |
        |                        | inverted_residual_block         | (None, 56, 56, 24)      |
        |------------------------+---------------------------------+-------------------------|
        | Stage 2                | inverted_residual_block (s=2)   | (None, 28, 28, 32)      |
        |                        | inverted_residual_block         | (None, 28, 28, 32)      |
        |                        | inverted_residual_block         | (None, 28, 28, 32)      |
        |------------------------+---------------------------------+-------------------------|
        | Stage 3                | inverted_residual_block (s=2)   | (None, 14, 14, 64)      |
        |                        | inverted_residual_block (x3)    | (None, 14, 14, 64)      |
        |                        | inverted_residual_block (x3)    | (None, 14, 14, 96)      |
        |------------------------+---------------------------------+-------------------------|
        | Stage 4                | inverted_residual_block (s=2)   | (None, 7, 7, 160)       |
        |                        | inverted_residual_block (x2)    | (None, 7, 7, 160)       |
        |                        | inverted_residual_block         | (None, 7, 7, 320)       |
        |                        | ConvolutionBlock (1x1, s=1)     | (None, 7, 7, 1280)      |
        |------------------------+---------------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling2D          | (None, 1280)            |
        |                        | fc (Logics)                     | (None, 1000)            |
         ------------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         --------------------------------------------
        |         Model Name       |    Params       |
        |--------------------------------------------|
        |     0.35 MobileNetV2     |    1,529,768    |
        |--------------------------------------------|
        |     0.5 MobileNetV2      |    1,987,224    |
        |--------------------------------------------|
        |     0.75 MobileNetV2     |    2,663,064    |
        |--------------------------------------------|
        |     1.0 MobileNetV2      |    3,538,984    |
        |--------------------------------------------|
        |     1.3 MobileNetV2      |    5,431,116    |
        |--------------------------------------------|
        |     1.4 MobileNetV2      |    6,156,440    |
         --------------------------------------------

    References:
        - Paper: “MobileNetV2: Inverted Residuals and Linear Bottlenecks”  
          https://arxiv.org/abs/1801.04381
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
    
        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
          https://github.com/tonylins/pytorch-mobilenet-v2

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, DepthwiseConv2D,
    Reshape, Dense, Dropout, GlobalAveragePooling2D,
    add, multiply, concatenate,
)

from .mobilenet_v1 import stem_block
from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    MLPBlock, DropPathV1, DropPathV2, LinearLayer,
    OperatorWrapper, UnstackWrapper,
)
from utils.model_processing import (
    process_model_input, correct_pad, create_layer_instance,
    validate_conv_arg, check_regularizer, create_model_backbone,
)
from utils.auxiliary_processing import make_divisible



def inverted_residual_block(
    inputs,
    filters,
    strides=(1, 1),
    expansion=1,
    alpha=1.,
    depth_multiplier=1,
    activation="relu6",
    normalizer="batch-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inverted_residual_block_{K.get_uid('inverted_residual_block')}"
        
    c = inputs.shape[-1]
    pointwise_filters = make_divisible(int(filters * alpha), 8)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)

    x = inputs

    if expansion > 1:
        x = Sequential([
            Conv2D(
                filters=expansion * c,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=False,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
        ], name=f"{name}.expand")(x)

    if strides == (2, 2):
        x = ZeroPadding2D(padding=correct_pad(x, 3), name=f"{name}.padding")(x)

    x = Sequential([
        DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=strides,
            padding='same' if strides == (1, 1) else 'valid',
            depth_multiplier=depth_multiplier,
            use_bias=False,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=pointwise_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.depthwise_conv")(x)

    if c == pointwise_filters and strides == (1, 1):
        return add([inputs, x], name=f"{name}.final")
    else:
        return LinearLayer(name=f"{name}.final")(x)
    

def MobileNet_v2(
    filters,
    alpha,
    depth_multiplier,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu6",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
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
        
    if isinstance(inputs, (tuple, list)):
        if len(inputs) == 3:
            height, width, _ = inputs
        elif len(inputs) == 4:
            _, height, width, _ = inputs
        else:
            raise ValueError("Invalid input shape tuple/list: expected 3 or 4 elements.")
    else:
        shape = inputs.shape
        if len(shape) != 4:
            raise ValueError("Input tensor must have rank 4: (batch, height, width, channels)")
        height, width = shape[1], shape[2]

    if weights == 'imagenet':
        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')
            
        if height != width and height not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '128, 160, 192 or 224 only.')
            
    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "alpha": alpha,
        "depth_multiplier": depth_multiplier,
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
        default_size=224,
        min_size=32,
        weights=weights
    )
    
    # Stage 0:
    x = create_layer_instance(
        stem_block,
        inputs=inputs,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stem"
    )

    # Stage 1:
    for i in range(3):
        x = create_layer_instance(
            inverted_residual_block,
            inputs=x,
            filters=int(filters * 3/4) if i != 0 else int(filters * 1/2),
            strides=(2, 2) if i == 0 else (1, 1),
            expansion=1 if i == 0 else 6,
            **layer_constant_dict,
            name=f"stage1.block{i + 2}"
        )
    
    # Stage 2:
    for i in range(3):
        x = create_layer_instance(
            inverted_residual_block,
            inputs=x,
            filters=filters,
            strides=(2, 2) if i == 0 else (1, 1),
            expansion=6,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    # Stage 3:
    for i in range(4):
        x = create_layer_instance(
            inverted_residual_block,
            inputs=x,
            filters=filters * 2,
            strides=(2, 2) if i == 0 else (1, 1),
            expansion=6,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )

    for i in range(3):
        x = create_layer_instance(
            inverted_residual_block,
            inputs=x,
            filters=filters * 3,
            strides=(1, 1),
            expansion=6,
            **layer_constant_dict,
            name=f"stage3.block{i + 5}"
        )

    # Stage 4:
    for i in range(3):
        x = create_layer_instance(
            inverted_residual_block,
            inputs=x,
            filters=filters * 5,
            strides=(2, 2) if i == 0 else (1, 1),
            expansion=6,
            **layer_constant_dict,
            name=f"stage4.block{i + 1}"
        )

    x = create_layer_instance(
        inverted_residual_block,
        inputs=x,
        filters=filters * 10,
        strides=(1, 1),
        expansion=6,
        **layer_constant_dict,
        name=f"stage4.block{i + 2}"
    )

    x = Sequential([
        Conv2D(
            filters=make_divisible(filters * 40 * alpha, 8) if alpha > 1.0 else filters * 40,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage4.block{i + 3}")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model = Model(inputs, x, name=f'{float(alpha)}-MobileNetV2-{height}')
    return model


def MobileNet_v2_backbone(
    filters,
    alpha,
    depth_multiplier,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu6",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=MobileNet_v2,
        custom_layers=custom_layers,
        filters=filters,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def MobileNet_base_v2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu6",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = MobileNet_v2(
        filters=32,
        alpha=1.,
        depth_multiplier=1,
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


def MobileNet_base_v2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu6",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=MobileNet_base_v2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    