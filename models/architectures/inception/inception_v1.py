"""
    InceptionV1: Multi-Branch Convolutional Backbone with Sparse Architecture
    
    Overview:
        InceptionV1 (also known as GoogLeNet) is a convolutional neural network
        architecture designed to achieve high accuracy with low computational cost
        through the use of multi-branch modules called Inception blocks.
    
        The core idea is to approximate a sparse CNN using dense components by
        performing multiple convolutions (1x1, 3x3, 5x5) and pooling in parallel,
        then concatenating their outputs. It emphasizes dimensionality reduction
        using 1x1 convolutions and deep supervision with auxiliary classifiers.
    
        Key innovations include:
            - Inception Module: Multi-path feature extraction with different receptive fields
            - 1x1 Convolution: Used for dimensionality reduction and increased efficiency
            - Auxiliary Classifiers: Added during training to improve gradient flow
    
    Key Components:
        • Inception Module:
            - A multi-branch block that applies:
                • 1x1 convolution
                • 1x1 → 3x3 convolution
                • 1x1 → 5x5 convolution
                • 3x3 max pooling → 1x1 convolution
            - All outputs are concatenated along the channel dimension.
            - Enables capturing features at multiple scales with minimal increase in computation.
    
        • Dimensionality Reduction (via 1x1 Conv):
            - 1x1 convolutions are used to reduce channel depth before expensive operations
              (like 3x3 and 5x5), reducing FLOPs and parameters.
            - Encourages network sparsity and feature reuse.
    
        • Auxiliary Classifiers:
            - Two classifiers are attached to intermediate layers during training.
            - Each includes a small convolutional head followed by average pooling,
              fully connected layers, and softmax.
            - They help combat vanishing gradients and act as regularizers.
    
        • Global Average Pooling:
            - Instead of fully connected layers at the end, InceptionV1 uses global average
              pooling to reduce overfitting and model size.
    
        • Stage-wise Structure:
            - The model is built in stages with increasing number of Inception modules.
            - Downsampling is done via stride-2 convolutions or pooling between stages.
    
    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 299, 299, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (7x7, s=2) | (None, 112, 112, 64)    |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 56, 56, 64)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (1x1, s=1) | (None, 56, 56, 64)      |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 56, 56, 192)     |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 28, 28, 192)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | inception_block_v1          | (None, 28, 28, 256)     |
        |                        | inception_block_v1          | (None, 28, 28, 480)     |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 14, 14, 480)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | inception_block_v1          | (None, 14, 14, 512)     |
        |                        | inception_block_v1          | (None, 14, 14, 512)     |
        |                        | inception_block_v1          | (None, 14, 14, 512)     |
        |                        | inception_block_v1          | (None, 14, 14, 528)     |
        |                        | inception_block_v1          | (None, 14, 14, 832)     |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 7, 7, 832)       |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | inception_block_v1          | (None, 7, 7, 832)       |
        |                        | inception_block_v1          | (None, 7, 7, 1024)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | AveragePooling2D (7x7, s=1) | (None, 1, 1, 1024)      |
        |                        | Flatten                     | (None, 1024)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         ------------------------------------------------------------------
        |        Model Name                               |     Params     |
        |-------------------------------------------------+----------------|
        |    Inception naive v1 (w/o auxiliary_logits)    |  104,273,800   |
        |    Inception naive v1 (auxiliary_logits)        |   97,525,688   |
        |-------------------------------------------------+----------------|
        |    Inception v1 (w/o auxiliary_logits)          |    6,991,272   |
        |    Inception v1 (auxiliary_logits)              |   13,370,744   |
         ------------------------------------------------------------------

    References:
        - Paper: “Going Deeper with Convolutions (Inception v1 / GoogLeNet)”  
          https://arxiv.org/abs/1409.4842
    
        - Original implementation (official Caffe version):  
          https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

        - TensorFlow/Keras implementation:
          https://github.com/guaiguaibao/GoogLeNet_Tensorflow2.0/tree/master/tensorflow2.0/GoogLeNet
          https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/GoogLeNet.py

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D,
    concatenate
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone



def inception_v1_naive_block(
    inputs,
    filters,
    use_bias=False,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    """
    Inception module, naive version

    :param inputs: input tensor.
    :param blocks:
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_naive_block_{K.get_uid('inception_v1_naive_block')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    
    # branch1
    branch1 = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch1")(inputs)

    # branch2
    branch2 = Sequential([
        Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch2")(inputs)

    # branch3
    branch3 = Sequential([
        Conv2D(
            filters=filters[2],
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch3")(inputs)

    # branch4
    branch4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding="same",
        name=f"{name}.branch4"
    )(inputs)

    merger = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return merger


def inception_v1_block(
    inputs,
    filters,
    use_bias=False,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    """
    Inception module with dimension reductions

    :param inputs: input tensor.
    :param blocks: filter block, respectively: #1×1, #3×3 reduce, #3×3, #5×5 reduce, #5×5, pool proj
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_block_{K.get_uid('inception_v1_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    # branch 1x1
    branch_1x1 = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch1")(inputs)

    # branch 3x3
    branch_3x3 = Sequential([
        Conv2D(
            filters=filters[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[2],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch2")(inputs)

    # branch 5x5
    branch_5x5 = Sequential([
        Conv2D(
            filters=filters[3],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[4],
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch3")(inputs)
    
    # branch pool
    branch_pool = Sequential([
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
        ),
        Conv2D(
            filters=filters[5],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch4")(inputs)

    merger = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis=-1, name=f"{name}.merger")
    return merger


def inception_v1_auxiliary_block(
    inputs,
    filters,
    use_bias=False,
    num_classes=1000,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    name=None
):
    """
    Inception auxiliary classifier module

    :param inputs: input tensor.
    :param num_classes: number off classes
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_auxiliary_block_{K.get_uid('inception_v1_auxiliary_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    return Sequential([
        AveragePooling2D(pool_size=5, strides=3),
        Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Flatten(),
        Dropout(rate=drop_rate),
        Dense(units=1024),
        get_activation_from_name(activation),
        Dense(units=1 if num_classes == 2 else num_classes),
        get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
    ], name=name)(inputs)


def GoogleNet(
    block,
    use_bias=False,
    auxiliary_logits=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
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
        
    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "use_bias": use_bias,
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
        min_size=64,
        weights=weights
    )

    # Stage 0
    x = Sequential([
        Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem.conv_block")(inputs)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="stem.pool")(x)
    
    # Stage 1
    x = Sequential([
        Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stage1.block1")(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="stage1.block2")(x)
    
    # Stage 2
    x = block(
        inputs=x,
        filters=[64, 128, 32] if block == inception_v1_naive_block else [64, 96, 128, 16, 32, 32],
        **layer_constant_dict,
        name="stage2.block1"
    )
    
    x = block(
        inputs=x,
        filters=[128, 192, 96] if block == inception_v1_naive_block else [128, 128, 192, 32, 96, 64],
        **layer_constant_dict,
        name="stage2.block2"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="stage2.pool")(x)

    # Stage 3
    x = block(
        inputs=x,
        filters=[192, 208, 48] if block == inception_v1_naive_block else [192, 96, 208, 16, 48, 64],
        **layer_constant_dict,
        name="stage3.block1"
    )

    if auxiliary_logits:
        aux1 = inception_v1_auxiliary_block(
            inputs=x,
            filters=128,
            num_classes=num_classes,
            **layer_constant_dict,
            name="stage3.auxiliary_logits1"
        )
        
    x = block(
        inputs=x,
        filters=[160, 224, 64] if block == inception_v1_naive_block else [160, 112, 224, 24, 64, 64],
        **layer_constant_dict,
        name="stage3.block2"
    )
    
    x = block(
        inputs=x,
        filters=[128, 256, 64] if block == inception_v1_naive_block else [128, 128, 256, 24, 64, 64],
        **layer_constant_dict,
        name="stage3.block3"
    )
    
    x = block(
        inputs=x,
        filters=[112, 288, 64] if block == inception_v1_naive_block else [112, 144, 288, 32, 64, 64],
        **layer_constant_dict,
        name="stage3.block4"
    )
                  
    if auxiliary_logits:
        aux2 = inception_v1_auxiliary_block(
            inputs=x,
            filters=128,
            num_classes=num_classes,
            **layer_constant_dict,
            name="stage3.auxiliary_logits2"
        )

    x = block(
        inputs=x,
        filters=[256, 320, 128] if block == inception_v1_naive_block else [256, 160, 320, 32, 128, 128],
        **layer_constant_dict,
        name="stage3.block5"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="stage3.pool")(x)

    # Stage 4
    x = block(
        inputs=x,
        filters=[256, 320, 128] if block == inception_v1_naive_block else [256, 160, 320, 32, 128, 128],
        **layer_constant_dict,
        name="stage4.block1"
    )
    
    x = block(
        inputs=x,
        filters=[384, 384, 128] if block == inception_v1_naive_block else [384, 192, 384, 48, 128, 128],
        **layer_constant_dict,
        name="stage4.block2"
    )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(7, 7), strides=(1, 1)),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    if auxiliary_logits:
        output = [aux1, aux2, x]
    else:
        output = x
        
    if block == inception_v1_naive_block:
        model_name = "Inception-naive-v1"
    elif block == inception_v1_block:
        model_name = "Inception-base-v1"
    else:
        model_name = "GoogleNet"

    model = Model(inputs=inputs, outputs=output, name=model_name)
    return model


def GoogleNet_backbone(
    block,
    use_bias=False,
    auxiliary_logits=False,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    return create_model_backbone(
        model_fn=GoogleNet,
        custom_layers=custom_layers,
        block=block,
        use_bias=use_bias,
        auxiliary_logits=auxiliary_logits,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_naive_v1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = GoogleNet(
        block=inception_v1_naive_block,
        use_bias=False,
        auxiliary_logits=False,
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


def Inception_naive_v1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    return create_model_backbone(
        model_fn=Inception_naive_v1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_base_v1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = GoogleNet(
        block=inception_v1_block,
        use_bias=False,
        auxiliary_logits=False,
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


def Inception_base_v1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    return create_model_backbone(
        model_fn=Inception_base_v1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
