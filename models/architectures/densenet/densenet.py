"""
    DenseNet: Backbone with Dense Connectivity and Feature Reuse
    
    Overview:
        DenseNet (Densely Connected Convolutional Network) is a convolutional
        neural network architecture known for its efficient feature reuse and
        strong gradient flow. It connects each layer to every other layer in a 
        feed-forward fashion within a dense block, allowing later layers to access 
        feature maps from all previous layers directly. DenseNet is widely used 
        in classification and object detection backbones due to its compact design 
        and representational strength.
    
        Key innovations include:
            - DenseBlock: Dense connectivity between layers for improved gradient flow
            - Transition Layer: Compression and downsampling between dense blocks
            - Growth Rate: Controls the number of feature maps added per layer
    
    Key Components:
        • DenseBlock:
            - A sequence of layers where each layer receives as input the concatenation
              of all preceding layers’ outputs within the block.
            - Encourages feature reuse and mitigates vanishing gradients.
            - Each layer typically consists of BatchNorm → ReLU → 1x1 Conv →
              BatchNorm → ReLU → 3x3 Conv.
    
        • Transition Layer:
            - Used between DenseBlocks to reduce the number of feature maps and spatial size.
            - Composed of BatchNorm → 1x1 Conv → 2x2 AvgPool.
            - The compression factor (usually < 1) controls the number of output channels.
    
        • Growth Rate:
            - Defines how many feature maps each layer in a DenseBlock adds.
            - A smaller growth rate results in a more compact model, while a larger rate
              increases capacity.
    
        • Bottleneck Layers (optional in DenseNet-BC):
            - 1x1 convolutions added before 3x3 conv to reduce input dimensionality.
            - Used to improve efficiency, especially for deeper DenseNets.
            
    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 224, 224, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ZeroPadding2D (3x3)         | (None, 230, 230, 3)     |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 112, 112, 64)    |
        |                        | ZeroPadding2D (1x1)         | (None, 114, 114, 64)    |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 56, 56, 64)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | dense_block (6x)            | (None, 56, 56, 256)     |
        |                        | transition_block            | (None, 28, 28, 128)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | dense_block (12x)           | (None, 28, 28, 512)     |
        |                        | transition_block            | (None, 14, 14, 256)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | dense_block (48x)           | (None, 14, 14, 1792)    |
        |                        | transition_block            | (None, 7, 7, 896)       |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | dense_block (32x)           | (None, 7, 7, 1920)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 1920)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         --------------------------------------
        |      Model Name     |    Params      |
        |--------------------------------------|
        |     DenseNet-121    |    8,062,504   |
        |---------------------|----------------|
        |     DenseNet-169    |   14,307,880   |
        |---------------------|----------------|
        |     DenseNet-201    |   20,242,984   |
        |---------------------|----------------|
        |     DenseNet-264    |   33,736,232   |
         --------------------------------------

    References:
        - Paper: “Densely Connected Convolutional Networks”  
          https://arxiv.org/abs/1608.06993
    
        - Original implementation (official Torch version):  
          https://github.com/liuzhuang13/DenseNet

        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
          https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
          https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Dense, MaxPooling2D, Dropout,
    AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
    concatenate
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone


def conv_block(
    inputs,
    growth_rate=32,
    scale_ratio=4,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A building block for a dense block

    :param inputs: input tensor.
    :param growth_rate: float, growth rate at dense layers.
    :param name: string, block label.
    :return: Output tensor for the block.
    """    
    if name is None:
        name = f"conv_block_{K.get_uid('conv_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=growth_rate * scale_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
    ], name=f"{name}.conv_block")(inputs)

    merge = concatenate([inputs, x], name=f"{name}.merger")
    return merge


def dense_block(
    inputs,
    blocks,
    growth_rate=32,
    scale_ratio=4,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A dense block.

    :param inputs: input tensor.
    :param blocks: integer, the number of building blocks.
    :param name: string, block label.
    :return: output tensor for the block.
    """
    if name is None:
        name = f"dense_block_{K.get_uid('dense_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    x = inputs
    for i in range(blocks):
        x = conv_block(
            inputs=x,
            growth_rate=growth_rate,
            scale_ratio=scale_ratio,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            name=f"{name}.step{str(i + 1)}",
        )
    return x


def transition_block(
    inputs,
    reduction,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A transition block.

    :param inputs: input tensor.
    :param reduction: float, compression rate at transition layers.
    :param name: string, block label.
    :return: output tensor for the block.
    """    
    if name is None:
        name = f"transition_block_{K.get_uid('transition_block')}"

    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=int(K.int_shape(inputs)[-1] * reduction),
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
    ], name=f"{name}.conv_block")(inputs)
    return x


def DenseNet(
    filters,
    num_blocks,
    growth_rate,
    scale_ratio,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
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
    x = Sequential([
        ZeroPadding2D(padding=[(3, 3), (3, 3)]),
        Conv2D(
            filters=filters,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        ZeroPadding2D(padding=[(1, 1), (1, 1)]),
    ], name="stem.block1")(inputs)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)

    for i, num_block in enumerate(num_blocks):
        block_name_prefix = f"stage{i + 1}"

        if num_block > 0:
            x = dense_block(
                inputs=x,
                blocks=num_block - 1 if num_block > 1 else num_block,
                growth_rate=growth_rate,
                scale_ratio=scale_ratio,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )

        if num_block > 1:
            if i != len(num_blocks) - 1:
                x = transition_block(
                    inputs=x,
                    reduction=0.5,
                    **layer_constant_dict,
                    name=f"{block_name_prefix}.block2"
                )
            else:
                x = Sequential([
                    get_normalizer_from_name(normalizer, epsilon=norm_eps),
                    get_activation_from_name(activation)
                ], name=f"{block_name_prefix}.block2")(x)
                 
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "DenseNet"
    if num_blocks == [7, 13, 25, 17]:
        model_name += "-121"
    elif num_blocks == [7, 13, 33, 33]:
        model_name += "-169"
    elif num_blocks == [7, 13, 49, 33]:
        model_name += "-201"
    elif num_blocks == [7, 13, 65, 49]:
        model_name += "-264"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DenseNet_backbone(
    filters,
    num_blocks,
    growth_rate,
    scale_ratio,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block1" if i == 0 else f"stage{i}.block1.step{j}.merger"
        for i, j in enumerate(num_blocks[:-1])
    ]
    
    return create_model_backbone(
        model_fn=DenseNet,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        growth_rate=growth_rate,
        scale_ratio=scale_ratio,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DenseNet121(
    inputs=[224, 224, 3],
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
    
    model = DenseNet(
        filters=64,
        num_blocks=[7, 13, 25, 17],
        growth_rate=32,
        scale_ratio=4,
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


def DenseNet121_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1.step6.merger",
        "stage2.block1.step12.merger",
        "stage3.block1.step24.merger",
    ]

    return create_model_backbone(
        model_fn=DenseNet121,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DenseNet169(
    inputs=[224, 224, 3],
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

    model = DenseNet(
        filters=64,
        num_blocks=[7, 13, 33, 33],
        growth_rate=32,
        scale_ratio=4,
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


def DenseNet169_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1.step6.merger",
        "stage2.block1.step12.merger",
        "stage3.block1.step32.merger",
    ]
    
    return create_model_backbone(
        model_fn=DenseNet169,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DenseNet201(
    inputs=[224, 224, 3],
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

    model = DenseNet(
        filters=64,
        num_blocks=[7, 13, 49, 33],
        growth_rate=32,
        scale_ratio=4,
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


def DenseNet201_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1.step6.merger",
        "stage2.block1.step12.merger",
        "stage3.block1.step48.merger",
    ]
    
    return create_model_backbone(
        model_fn=DenseNet201,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DenseNet264(
    inputs=[224, 224, 3],
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

    model = DenseNet(
        filters=64,
        num_blocks=[7, 13, 65, 49],
        growth_rate=32,
        scale_ratio=4,
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


def DenseNet264_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1.step6.merger",
        "stage2.block1.step12.merger",
        "stage3.block1.step64.merger",
    ]

    return create_model_backbone(
        model_fn=DenseNet264,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
