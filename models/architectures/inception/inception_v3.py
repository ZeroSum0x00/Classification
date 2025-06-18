"""
    InceptionV3: Enhanced Multi-Branch Backbone with Factorized Convolutions and Label Smoothing
    
    Overview:
        InceptionV3 is an improved version of the original Inception (GoogLeNet)
        architecture, focusing on improving both performance and efficiency
        through several design innovations.
    
        It refines the Inception module by factorizing convolutions (e.g., 3x3 into 1x3 + 3x1),
        introducing auxiliary losses, label smoothing regularization, and batch normalization.
        The architecture is deeper, wider, and more optimized for training and inference speed.
    
        Key innovations include:
            - Factorized Convolutions: Reduces computation while preserving receptive field
            - Grid Size Reduction: Efficient downsampling without feature loss
            - Auxiliary Classifiers: Used for deep supervision and regularization
    
    Key Components:
        • Inception Module v3:
            - An evolution of the original Inception block with factorized convolutions:
                • 3×3 replaced with 1×3 + 3×1
                • 5×5 replaced with two 3×3
            - Multi-branch processing:
                • 1×1 conv
                • 1×1 → (1×3 + 3×1) conv
                • 1×1 → two 3×3 convs
                • 3×3 max pool → 1×1 conv
            - Outputs from all branches are concatenated.
    
        • Factorized Convolutions:
            - High-cost convolutions (e.g., 7×7) are broken into smaller kernels to reduce FLOPs.
            - Example: A 7×7 convolution becomes a 1×7 followed by a 7×1.
    
        • Grid Size Reduction:
            - Replaces stride-2 pooling with more structured downsampling:
                • Parallel 3×3 stride-2 conv
                • 3×3 conv followed by 3×3 stride-2 conv
                • 3×3 stride-2 max pooling
            - Merges all branches for efficient spatial resolution reduction.
    
        • Auxiliary Classifiers:
            - Intermediate outputs used during training for better gradient flow.
            - Each auxiliary head includes average pooling → 1x1 conv → FC layers → softmax.
    
        • Label Smoothing & Regularization:
            - Adds a small probability to incorrect classes to reduce overconfidence.
            - Improves generalization and robustness of predictions.
    
        • Batch Normalization:
            - Applied throughout the network to accelerate training convergence and stability.
    
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
         --------------------------------------
        |     Model Name     |    Params       |
        |--------------------------------------|
        |    Inception v3    |   23,869,000    |
         --------------------------------------

    References:
        - Paper: “Rethinking the Inception Architecture for Computer Vision”
          https://arxiv.org/abs/1512.00567
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py
    
        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    concatenate,
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone


def convolution_block(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    use_bias=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"convolution_block_{K.get_uid('convolution_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    return Sequential([
        Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=name)(inputs)


def stem_block(
    inputs,
    filters,
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
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block1"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block2"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block3"
    )
    
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stem.pooling"
    )(x)

    x = convolution_block(
        inputs=x,
        filters=int(filters * 5/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stage1.block1"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stage1.block2"
    )
    
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stage1.block3"
    )(x)
    return x


def inception_block_A(
    inputs,
    filters_branch1,
    filters_branch2,
    filters_branch3,
    filters_branch4,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_a_{K.get_uid('inception_block_a')}"

    filters_branch1 = [filters_branch1] if isinstance(filters_branch1, int) else filters_branch1
    filters_branch4 = [filters_branch4] if isinstance(filters_branch4, int) else filters_branch4
    regularizer_decay = check_regularizer(regularizer_decay)

    # Branch 1: 1x1 Convolution
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters_branch1[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1"
    )

    # Branch 2: 1x1 Convolution -> 5x5 Convolution
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters_branch2[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[1],
        kernel_size=(5, 5),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv2"
    )

    # Branch 3: 1x1 -> 3x3 -> 3x3 Convolutions
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters_branch3[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[1],
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[2],
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )

    # Branch 4: AvgPooling -> 1x1 Convolution
    branch4 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"{name}.branch4.pool"
    )(inputs)
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters_branch4[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv"
    )

    # Concatenate all branches
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return output


def inception_block_B(
    inputs,
    filters_branch1,
    filters_branch2,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_b_{K.get_uid('inception_block_b')}"

    filters_branch1 = [filters_branch1] if isinstance(filters_branch1, int) else filters_branch1
    regularizer_decay = check_regularizer(regularizer_decay)

    # Branch 1: 3x3 Convolution with stride 2
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters_branch1[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1"
    )

    # Branch 2: 1x1 -> 3x3 -> 3x3 Convolutions
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters_branch2[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[1],
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[2],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv3"
    )

    # Branch 3: MaxPooling
    branch3 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch3.pool"
    )(inputs)

    # Concatenate all branches
    output = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return output

    
def inception_block_C(
    inputs,
    filters_branch1,
    filters_branch2,
    filters_branch3,
    filters_branch4,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_c_{K.get_uid('inception_block_c')}"

    filters_branch1 = [filters_branch1] if isinstance(filters_branch1, int) else filters_branch1
    filters_branch4 = [filters_branch4] if isinstance(filters_branch4, int) else filters_branch4
    regularizer_decay = check_regularizer(regularizer_decay)

    # Branch 1: 1x1 Convolution
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters_branch1[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1"
    )

    # Branch 2: 1x1 -> 1x7 -> 7x1 Convolutions
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters_branch2[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[1],
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[2],
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv3"
    )

    # Branch 3: 1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7 Convolutions
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters_branch3[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[1],
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[2],
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[3],
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv4"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters_branch3[4],
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv5"
    )

    # Branch 4: AvgPooling -> 1x1 Convolution
    branch4 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"{name}.branch4.pool"
    )(inputs)
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters_branch4[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv"
    )

    # Concatenate all branches
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return output


def inception_block_D(
    inputs,
    filters_branch1,
    filters_branch2,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_d_{K.get_uid('inception_block_d')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    # Branch 1: 1x1 conv -> 3x3 conv with stride 2
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters_branch1[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        name=f"{name}.branch1.conv1"
    )
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=filters_branch1[1],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name=f"{name}.branch1.conv2"
    )

    # Branch 2: 1x1 -> 1x7 -> 7x1 -> 3x3 stride 2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters_branch2[0],
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        name=f"{name}.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[1],
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=False,
        name=f"{name}.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[2],
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=False,
        name=f"{name}.branch2.conv3"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters_branch2[3],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name=f"{name}.branch2.conv4"
    )

    # Branch 3: MaxPooling with stride 2
    branch3 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch3.pool"
    )(inputs)

    # Concatenate all branches
    output = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return output


def inception_block_E(
    inputs,
    filters_branch1,
    filters_branch2,
    filters_branch3,
    filters_branch4,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_e_{K.get_uid('inception_block_e')}"

    filters_branch1 = [filters_branch1] if isinstance(filters_branch1, int) else filters_branch1
    filters_branch4 = [filters_branch4] if isinstance(filters_branch4, int) else filters_branch4
    regularizer_decay = check_regularizer(regularizer_decay)

    # Branch 1x1
    branch1x1 = convolution_block(
        inputs=inputs,
        filters=filters_branch1[0],
        kernel_size=(1, 1),
        use_bias=False,
        name=f"{name}.branch1.conv"
    )

    # Branch 3x3 (split into 1x3 and 3x1)
    branch3x3 = convolution_block(
        inputs=inputs,
        filters=filters_branch2[0],
        kernel_size=(1, 1),
        use_bias=False,
        name=f"{name}.branch2.conv1"
    )
    
    branch3x3_1 = convolution_block(
        inputs=branch3x3,
        filters=filters_branch2[1],
        kernel_size=(1, 3),
        use_bias=False,
        name=f"{name}.branch2.conv2a"
    )
    
    branch3x3_2 = convolution_block(
        inputs=branch3x3,
        filters=filters_branch2[2],
        kernel_size=(3, 1),
        use_bias=False,
        name=f"{name}.branch2.conv2b"
    )
    
    branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=-1, name=f"{name}.branch2.merger")

    # Branch 3x3 double (split again into 1x3 and 3x1)
    branch3x3dbl = convolution_block(
        inputs=inputs,
        filters=filters_branch3[0],
        kernel_size=(1, 1),
        use_bias=False,
        name=f"{name}.branch3.conv1"
    )
    
    branch3x3dbl = convolution_block(
        inputs=branch3x3dbl,
        filters=filters_branch3[1],
        kernel_size=(3, 3),
        use_bias=False,
        name=f"{name}.branch3.conv2"
    )
    
    branch3x3dbl_1 = convolution_block(
        inputs=branch3x3dbl,
        filters=filters_branch3[2],
        kernel_size=(1, 3),
        use_bias=False,
        name=f"{name}.branch3.conv3a"
    )
    
    branch3x3dbl_2 = convolution_block(
        inputs=branch3x3dbl,
        filters=filters_branch3[3],
        kernel_size=(3, 1),
        use_bias=False,
        name=f"{name}.branch3.conv3b"
    )
    
    branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=-1, name=f"{name}.branch3.merger")

    # Branch Pool
    branch_pool = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name=f"{name}.branch4.pool"
    )(inputs)
    
    branch_pool = convolution_block(
        inputs=branch_pool,
        filters=filters_branch4[0],
        kernel_size=(1, 1),
        use_bias=False,
        name=f"{name}.branch4.conv"
    )

    # Concatenate all branches
    x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name=f"{name}.merger")
    return x

        
def Inception_v3(
    num_blocks,
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

    x = stem_block(
        inputs=inputs,
        filters=32,
        **layer_constant_dict,
        name="stem"
    )

    for i in range(num_blocks[0]):
        x = inception_block_A(
            inputs=x,
            filters_branch1=64,
            filters_branch2=[48, 64],
            filters_branch3=[64, 96, 96],
            filters_branch4=32 if i == 0 else 64,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = inception_block_B(
        inputs=x,
        filters_branch1=384,
        filters_branch2=[64, 96, 96],
        name=f"stage2.block1"
    )

    for i in range(num_blocks[1]):
        if i == 0:
            filters_branch2 = [128, 128, 192]
            filters_branch3 = [128, 128, 128, 128, 192]
        elif i == num_blocks[1] - 1:
            filters_branch2 = [192, 192, 192]
            filters_branch3 = [192, 192, 192, 192, 192]
        else:
            filters_branch2 = [160, 160, 192]
            filters_branch3 = [160, 160, 160, 160, 192]
            
        x = inception_block_C(
            inputs=x,
            filters_branch1=192,
            filters_branch2=filters_branch2,
            filters_branch3=filters_branch3,
            filters_branch4=192,
            name=f"stage2.block{i + 2}"
        )

    x = inception_block_D(
        inputs=x,
        filters_branch1=[192, 320],
        filters_branch2=[192, 192, 192, 192],
        name=f"stage3.block1"
    )

    # mixed 9: 8 x 8 x 2048
    for i in range(num_blocks[2]):
        x = inception_block_E(
            inputs=x,
            filters_branch1=320,
            filters_branch2=[384, 384, 384],
            filters_branch3=[448, 384, 384, 384],
            filters_branch4=192,
            name=f"stage3.block{i + 2}"
        )

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    if num_blocks == [3, 4, 2]:
        model_name = "Inception-base-v3"
    else:
        model_name = "Inception-v3"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Inception_v3_backbone(
    num_blocks,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        f"stage1.block{num_blocks[0]}.merger",
        f"stage2.block{num_blocks[1] + 1}.merger",
    ]
    
    return create_model_backbone(
        model_fn=Inception_v3,
        custom_layers=custom_layers,
        num_blocks=num_blocks,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_base_v3(
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
    
    model = Inception_v3(
        num_blocks=[3, 4, 2],
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


def Inception_base_v3_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage1.block3.merger",
        "stage2.block5.merger",
    ]

    return create_model_backbone(
        model_fn=Inception_base_v3,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
