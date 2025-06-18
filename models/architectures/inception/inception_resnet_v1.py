"""
    Inception ResNet V1: Residual Multi-Branch Backbone Combining Inception and ResNet
    
    Overview:
        Inception-ResNet-v1 is a hybrid architecture that combines the multi-branch
        feature extraction capability of Inception modules with the fast convergence
        and training stability of residual connections from ResNet.
    
        This network inherits the modular structure of Inception-v3 while integrating
        shortcut (residual) paths in each block. It achieves high accuracy with efficient
        training speed and is widely used in tasks like face recognition and transfer learning.
    
        Key innovations include:
            - Residual Inception Modules: Fused Inception blocks with identity shortcuts
            - Scalable Architecture: Repeated residual blocks with clear stage boundaries
            - Fast Convergence: Enabled by ResNet-style residual learning
    
    Key Components:
        • Stem Block:
            - Input processing block that downsamples and expands early features.
            - Uses 3x3 convolutions and multi-branch conv/max-pool paths.
            - Produces rich low-level features and reduces resolution.
    
        • Inception-ResNet Blocks:
            - All blocks follow:  
              **Branching paths (Inception-style) → Concatenate → 1×1 Conv → Residual Add**
            - Types of blocks:
                - **Block35 (Inception-ResNet-A)** for 35×35 resolution
                    • Uses 1x1 and 3x3 conv branches, plus 3x3 → 3x3 stacked convs
                - **Block17 (Inception-ResNet-B)** for 17×17 resolution
                    • Uses asymmetric convolutions (1x7, 7x1)
                - **Block8 (Inception-ResNet-C)** for 8×8 resolution
                    • Includes 1x1 and 3x3 convs
            - Each block ends with 1×1 conv for dimensionality adjustment before residual addition.
    
        • Scaling of Residuals:
            - A small scaling factor (e.g., 0.1) is applied to the residual before addition.
            - This stabilizes training when combining high-dimensional inception outputs with residuals.
    
        • Reduction Blocks:
            - **Reduction-A**: Reduces from 35×35 to 17×17 using stride-2 convs and pooling.
            - **Reduction-B**: Reduces from 17×17 to 8×8 using deeper conv branches.
            - Similar to grid reduction in Inception-v3/v4.
    
        • Global Average Pooling and Classifier:
            - After residual blocks, global average pooling reduces spatial size.
            - Followed by dropout and final fully connected layer for classification.
    
    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 299, 299, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=2) | (None, 149, 149, 32)    |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 147, 147, 32)    |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 145, 145, 64)    |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 72, 72, 64)      |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 72, 72, 80)      |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 70, 70, 192)     |
        |                        | ConvolutionBlock (3x3, s=2) | (None, 35, 35, 256)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | inception_resnet_A (5x)     | (None, 35, 35, 256)     |
        |                        | reduction_A                 | (None, 17, 17, 896)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | inception_resnet_B (10x)    | (None, 17, 17, 896)     |
        |                        | reduction_B                 | (None, 8, 8, 1792)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | inception_resnet_C (5x)     | (None, 8, 8, 1792)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | AveragePooling2D (8x8, s=1) | (None, 1, 1, 1792)      |
        |                        | Flatten                     | (None, 1792)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
         
         ---------------------------------------------
        |         Model Name        |     Params      |
        |---------------------------------------------|
        |    Inception Resnet v1    |    21,684,184   |
         ---------------------------------------------

    References:
        - Paper: “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning”  
          https://arxiv.org/abs/1602.07261
    
        - TensorFlow/Keras implementation:
          https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v2.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Lambda, Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    concatenate, add,
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import (
    process_model_input, validate_conv_arg,
    check_regularizer, create_model_backbone,
)



def convolution_block(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    name=None
):
    if name is None:
        name = f"convolution_block_{K.get_uid('convolution_block')}"

    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)

    return Sequential([
        Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
    ], name=f"{name}.conv_block")(inputs)


def stem_block(
    inputs,
    filters=32,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"stem_block_{K.get_uid('stem_block')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    
    x = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv1"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv2"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv3"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f"{name}.pool")(x)
    
    x = convolution_block(
        inputs=x,
        filters=int(filters * 5/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv4"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv5"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 8,
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv6"
    )
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ")(x)
    return x


def inception_resnet_A(
    inputs,
    filters=32,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_a_{K.get_uid('inception_resnet_a')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    
    shortcut = inputs

    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step3"
    )
    
    # merger
    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 8,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)
        
    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def reduction_A(
    inputs,
    k=192,
    l=192,
    m=256,
    n=384,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"reduction_a_{K.get_uid('reduction_a')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    # branch1
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=n,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2"
    )

    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=k,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=l,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=m,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step3"
    )

    # merger
    out = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def inception_resnet_B(
    inputs,
    filters=128,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_b_{K.get_uid('inception_resnet_b')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    shortcut = inputs

    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )
    
    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step3"
    )

    # merger
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 7,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)

    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def reduction_B(
    inputs,
    filters=256,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"reduction_b_{K.get_uid('reduction_b')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    # branch1
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)
    
    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )

    # branch4
    branch4 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step3"
    )

    # merger
    out = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def inception_resnet_C(
    inputs,
    filters=128,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_c_{K.get_uid('inception_resnet_c')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
        
    shortcut = inputs
    
    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=int(filters * 3/2),
        kernel_size=(1, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=int(filters * 3/2),
        kernel_size=(3, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step3"
    )

    # merger
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 14,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)
        
    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def Inception_Resnet_v1(
    filters,
    num_blocks,
    scale_residual,
    inputs=[299, 299, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
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
        default_size=299,
        min_size=64,
        weights=weights
    )

    # Stem block
    x = stem_block(
        inputs=inputs,
        filters=filters,
        **layer_constant_dict,
        name="stem"
    )

    # Inception-A
    for i in range(num_blocks[0]):
        x = inception_resnet_A(
            inputs=x,
            filters=filters,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    # Reduction-A
    x = reduction_A(
        inputs=x,
        k=192,
        l=192,
        m=256,
        n=384,
        **layer_constant_dict,
        name=f"stage1.block{i + 2}"
    )

    # Inception-B
    for i in range(num_blocks[1]):
        x = inception_resnet_B(
            inputs=x,
            filters=filters * 4,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    # Reduction-B
    x = reduction_B(
        inputs=x,
        filters=filters * 8,
        **layer_constant_dict,
        name=f"stage2.block{i + 2}"
    )
                  
    # Inception-C
    for i in range(num_blocks[2]):
        x = inception_resnet_C(
            inputs=x,
            filters=filters * 4,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(8, 8), strides=(1, 1)),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    if filters == 32 and num_blocks == [5, 10, 5] and scale_residual:
        model_name = "Inception-Resnet-base-v1"
    else:
        model_name = "Inception-Resnet-v1"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Inception_Resnet_v1_backbone(
    filters,
    num_blocks,
    scale_residual,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.conv5.conv_block",
        f"stage1.block{num_blocks[0] + 1}.branch3.step2.conv_block",
        f"stage2.block{num_blocks[1] + 1}.branch4.step2.conv_block",
    ]
    
    return create_model_backbone(
        model_fn=Inception_Resnet_v1,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        scale_residual=scale_residual,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_Resnet_base_v1(
    inputs=[299, 299, 3],
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
) -> Model:
    
    model = Inception_Resnet_v1(
        filters=32,
        num_blocks=[5, 10, 5],
        scale_residual=True,
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


def Inception_Resnet_base_v1_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.conv5.conv_block",
        "stage1.block6.branch3.step2.conv_block",
        "stage2.block11.branch4.step2.conv_block",
    ]

    return create_model_backbone(
        model_fn=Inception_Resnet_base_v1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
