"""
    MobileNetV1: Lightweight CNN Backbone with Depthwise Separable Convolutions
    
    Overview:
        MobileNetV1 is a compact and efficient convolutional neural network (CNN)
        backbone designed for mobile and embedded devices. It introduces **depthwise
        separable convolutions**, a factorized form of standard convolutions that
        significantly reduces computational cost while preserving accuracy.
    
        Key innovations include:
            - Depthwise Separable Convolution: Decomposes standard conv into depthwise + pointwise
            - Width Multiplier (α): Controls number of channels for model size adjustment
            - Resolution Multiplier (ρ): Reduces input image resolution for further speedup
    
    Key Components:
        • Standard Convolution vs. Depthwise Separable Convolution:
            - **Standard Conv**: Applies `K×K×Cin` filters for each output channel → expensive
            - **Depthwise Separable Conv**: Splits into two steps:
                1. **Depthwise Convolution**:
                    - Applies a single `K×K` filter per input channel (no cross-channel mixing)
                2. **Pointwise Convolution (1×1)**:
                    - Mixes features across channels after depthwise step
            - Reduces computation by ~8–9× compared to standard conv with same parameters
    
        • MobileNet Block:
            - Each block is composed of:
                - DepthwiseConv → BatchNorm → ReLU
                - PointwiseConv → BatchNorm → ReLU
            - No residual connections (unlike later MobileNet versions)

        • Width Multiplier (α):
            - Scales the number of channels per layer
            - Typical values: α ∈ {1.0, 0.75, 0.5, 0.25}
            - Enables trade-off between latency and accuracy
    
        • Resolution Multiplier (ρ):
            - Scales down input image size (e.g., 224×224 → 160×160)
            - Further reduces FLOPs, useful for low-resource environments
    
        • No Residual Connections:
            - Unlike ResNet or MobileNetV2, MobileNetV1 uses a plain feedforward stack

    General Model Architecture:
         ------------------------------------------------------------------------------------
        | Stage                  | Layer                           | Output Shape            |
        |------------------------+---------------------------------+-------------------------|
        | Input                  | input_layer                     | (None, 224, 224, 3)     |
        |------------------------+---------------------------------+-------------------------|
        | Stem                   | ZeroPadding2D (1x1)             | (None, 225, 225, 3)     |
        |                        | ConvolutionBlock (3x3, s=2)     | (None, 112, 112, 32)    |
        |------------------------+---------------------------------+-------------------------|
        | Stage 1                | depthwise_separable_block (s=2) | (None, 112, 112, 64)    |
        |------------------------+---------------------------------+-------------------------|
        | Stage 2                | depthwise_separable_block (s=2) | (None, 56, 56, 128)     |
        |                        | depthwise_separable_block       | (None, 56, 56, 128)     |
        |------------------------+---------------------------------+-------------------------|
        | Stage 3                | depthwise_separable_block (s=2) | (None, 28, 28, 256)     |
        |                        | depthwise_separable_block       | (None, 28, 28, 256)     |
        |------------------------+---------------------------------+-------------------------|
        | Stage 4                | depthwise_separable_block (s=2) | (None, 14, 14, 512)     |
        |                        | depthwise_separable_block (x5)  | (None, 14, 14, 512)     |
        |------------------------+---------------------------------+-------------------------|
        | Stage 5                | depthwise_separable_block (s=2) | (None, 7, 7, 1024)      |
        |                        | depthwise_separable_block (x5)  | (None, 7, 7, 1024)      |
        |------------------------+---------------------------------+-------------------------|
        | CLS Logics             | AveragePooling2D (7x7, s=1)     | (None, 1, 1, 1024)      |
        |                        | Conv2D (1x1, s=1)               | (None, 1, 1, 1000)      |
        |                        | Flatten                         | (None, 1000)            |
         ------------------------------------------------------------------------------------

    Model Parameter Comparison:
         --------------------------------------------
        |         Model Name       |    Params       |
        |--------------------------------------------|
        |     0.25 MobileNetV1     |      475,544    |
        |--------------------------------------------|
        |     0.5 MobileNetV1      |    1,342,536    |
        |--------------------------------------------|
        |     0.75 MobileNetV1     |    2,601,976    |
        |--------------------------------------------|
        |     1.0 MobileNetV1      |    4,253,864    |
         --------------------------------------------
    
    References:
        - Paper: “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications”  
          https://arxiv.org/abs/1704.04861
    
        - Official TensorFlow implementation:  
          https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet_v1

        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py
          
        - PyTorch version (community):  
          https://github.com/kuan-wang/pytorch-mobilenet-v1

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, DepthwiseConv2D,
    Dropout, AveragePooling2D, Flatten,
    add, multiply, concatenate,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    MLPBlock, DropPathV1, DropPathV2, LinearLayer,
    OperatorWrapper, UnstackWrapper,
)
from utils.model_processing import (
    process_model_input, correct_pad, validate_conv_arg,
    check_regularizer, create_model_backbone,
)



def stem_block(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(2, 2),
    alpha=1,
    activation="relu6",
    normalizer="batch-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"stem_block_{K.get_uid('stem_block')}"
        
    filters = int(filters * alpha)
    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)

    return Sequential([
        ZeroPadding2D(padding=correct_pad(inputs, 3)),
        Conv2D(
            filters=filters, 
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=name)(inputs)


def depthwise_separable_convolutional(
    inputs,
    filters,
    strides=(1, 1),
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
        name = f"depthwise_separable_convolutional_{K.get_uid('depthwise_separable_convolutional')}"
        
    pointwise_filters = int(filters * alpha)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        ZeroPadding2D(
            padding=[(0, 1), (0, 1)]
        )
        if tuple(strides) != (1, 1) else LinearLayer(),
        DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=strides,
            padding="valid" if tuple(strides) != (1, 1) else "same",
            depth_multiplier=depth_multiplier,
            use_bias=False,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=pointwise_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=name)(inputs)
    return x
    

def MobileNet_v1(
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

    if weights == "imagenet":
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
    x = stem_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        alpha=alpha,
        **layer_constant_dict,
        name="stem"
    )

    # Stage 1:
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 2,
        strides=(1, 1),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage1.block1"
    )

    # Stage 2:
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 4,
        strides=(2, 2),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage2.block1"
    )
        
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 4,
        strides=(1, 1),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage2.block2"
    )
        
    # Stage 3:
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 8,
        strides=(2, 2),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage3.block1"
    )
        
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 8,
        strides=(1, 1),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage3.block2"
    )
        
    # Stage 4:
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 16,
        strides=(2, 2),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage4.block1"
    )
        
    for i in range(5):
        x = depthwise_separable_convolutional(
            inputs=x,
            filters=filters * 16,
            strides=(1, 1),
            alpha=alpha,
            depth_multiplier=depth_multiplier,
            **layer_constant_dict,
            name=f"stage4.block{i + 2}"
        )

    # Stage 5:
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 32,
        strides=(2, 2),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage5.block1"
    )
        
    x = depthwise_separable_convolutional(
        inputs=x,
        filters=filters * 32,
        strides=(1, 1),
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        **layer_constant_dict,
        name="stage5.block2"
    )

    if include_head:
        out_dim = 1 if num_classes == 2 else num_classes
        x = Sequential([
            AveragePooling2D(pool_size=(7, 7), strides=(1, 1)),
            Dropout(rate=drop_rate),
            Conv2D(
                filters=out_dim, 
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
            ),
            Flatten(),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model = Model(inputs, x, name=f"{float(alpha)}-MobileNetV1-{height}")
    return model


def MobileNet_v1_backbone(
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
        "stage1.block1",
        "stage2.block2",
        "stage3.block2",
        "stage4.block6",
    ]

    return create_model_backbone(
        model_fn=MobileNet_v1,
        custom_layers=custom_layers,
        filters=filters,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def MobileNet_base_v1(
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
    
    model = MobileNet_v1(
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


def MobileNet_base_v1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu6",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block1",
        "stage2.block2",
        "stage3.block2",
        "stage4.block6",
    ]

    return create_model_backbone(
        model_fn=MobileNet_base_v1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
