"""
    InceptionV4: Deep Multi-Path Backbone with Residual-Free Optimization and Structured Scaling
    
    Overview:
        InceptionV4 is a deeper and more powerful evolution of the Inception family,
        combining improved inception modules with better regularization, deeper stages,
        and more structured grid reduction strategies.
    
        Unlike Inception-ResNet, InceptionV4 avoids residual connections and relies on
        pure Inception blocks. It adopts standardized and modularized block types to
        allow more scalable and tunable architectures. This design improves representational
        capacity without overly increasing computational cost.
    
        Key innovations include:
            - Pure Inception Blocks (A, B, C): Modular, stackable inception variants
            - Stem Block: Optimized front-end block for early feature extraction
            - Grid Reduction Blocks: Efficient downsampling with minimal information loss
    
    Key Components:
        • Stem Block:
            - The input stage of InceptionV4, responsible for early spatial reduction.
            - Includes multiple 3x3 convolutions and branching paths (e.g., 3x3 max pool,
              3x3 stride-2 conv, and 1x1 → 3x3 conv).
            - Outputs rich feature maps while reducing resolution from the input image.
    
        • Inception-A / B / C Blocks:
            - **Inception-A**: Optimized for low-resolution feature maps (35×35).
                • 1x1 conv, 3x3 conv, 3x3 → 3x3 stacked convs, avg pooling branches.
            - **Inception-B**: Used for medium-resolution feature maps (17×17).
                • Extensive use of factorized 7x7 convolutions (1x7 + 7x1).
            - **Inception-C**: Designed for low-resolution maps (8×8).
                • Incorporates asymmetric convolutions and expansion via 1x1 convs.
            - All blocks use multi-branch design and concatenate the outputs.
    
        • Grid Reduction Blocks (Reduction-A / B):
            - Replaces max pooling with structured downsampling paths:
                • Parallel 3x3 stride-2 conv
                • Conv paths with increasing depth and stride
                • 3x3 stride-2 max pooling
            - Used between stages to halve feature map resolution (e.g., 35→17, 17→8).
    
        • Stage-wise Structure:
            - InceptionV4 follows a clear block pattern:
            - Encourages regularity, tuning flexibility, and scalable depth.
    
        • Global Average Pooling:
            - At the end of the network, a GAP layer replaces fully connected layers to reduce overfitting.
            - Final FC + Softmax layer for classification.

    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 299, 299, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=2) | (None, 149, 149, 32)    |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 147, 147, 32)    |
        |                        | ConvolutionBlock (3x3, s=1) | (None, 147, 147, 64)    |
        |                        | double_branch + concat      | (None, 73, 73, 160)     |
        |                        | double_branch + concat      | (None, 71, 71, 192)     |
        |                        | double_branch + concat      | (None, 35, 35, 384)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | inception_A (4x)            | (None, 35, 35, 384)     |
        |                        | reduction_A                 | (None, 17, 17, 1024)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | inception_B (7x)            | (None, 17, 17, 1024)    |
        |                        | reduction_B                 | (None, 8, 8, 1536)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | inception_C (3x)            | (None, 8, 8, 1536)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | AveragePooling2D (8x8, s=1) | (None, 1, 1, 1536)      |
        |                        | Flatten                     | (None, 1536)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         --------------------------------------------
        |        Model Name        |     Params      |
        |--------------------------------------------|
        |       Inception v4       |    42,742,984   |
         --------------------------------------------

    References:
        - Paper: “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning”  
          https://arxiv.org/abs/1602.07261
    
        - TensorFlow/Keras implementation:
          https://github.com/titu1994/Inception-v4/blob/master/inception_v4.py
          https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
    concatenate
)

from .inception_v3 import convolution_block
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone


def stem_block(
    inputs,
    filters,
    use_bias=False,
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
        padding='valid',
        use_bias=use_bias,
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
        padding='valid',
        use_bias=use_bias,
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
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block3"
    )

    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stem.block4.branch1.pooling"
    )(x)
    
    branch2 = convolution_block(
        inputs=x,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block4.branch2.conv"
    )
    
    x = concatenate([branch1, branch2], axis=-1, name='stem.merger1')
    
    branch1 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch1.conv1"
    )
    
    branch1 = convolution_block(
        branch1,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch1.conv2"
    )

    branch2 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv3"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv4"
    )

    x = concatenate([branch1, branch2], axis=-1, name='stem.merger2')

    branch1 = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block6.branch1.conv"
    )
    
    branch2 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stem.block6.branch2.pooling"
    )(x)
    
    x = concatenate([branch1, branch2], axis=-1, name='stem.merger3')
    return x


def inception_A(
    inputs,
    filters,
    use_bias=False,
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

    regularizer_decay = check_regularizer(regularizer_decay)
    
    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )

    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )

    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f'{name}.merger')
    return x


def reduction_A(
    inputs,
    k=192,
    l=224,
    m=256,
    n=384,
    use_bias=False,
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
    
    # Branch 1:
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)
    
    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=n,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=k,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=l,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=m,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step3"
    )

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return x


def inception_B(
    inputs,
    filters,
    use_bias=False,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_b_{K.get_uid('inception_b')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    
    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters * 3,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=int(filters * 7/4),
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=filters * 2,
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 3/2),
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 7/4),
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 7/4),
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv4"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters * 2,
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv5"
    )
    
    x = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return x


def reduction_B(
    inputs,
    filters,
    use_bias=False,
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
        
    # Branch 1:
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name=f"{name}.branch1.pooling"
    )(inputs)

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv2"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=int(filters * 4/3),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=int(filters * 4/3),
        kernel_size=(1, 7),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=int(filters * 5/3),
        kernel_size=(7, 1),
        strides=(1, 1),
        use_bias=use_bias,
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
        filters=int(filters * 5/3),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv4"
    )

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return x


def inception_C(
    inputs,
    filters,
    use_bias=False,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_c_{K.get_uid('inception_c')}"

    regularizer_decay = check_regularizer(regularizer_decay)
        
    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )
    
    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )
    

    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch31 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(1, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch32 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 7/4),
        kernel_size=(1, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters * 2,
        kernel_size=(3, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )
    
    branch41 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(1, 3),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv4"
    )
    
    branch42 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(3, 1),
        strides=(1, 1),
        use_bias=use_bias,
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv5"
    )
    
    x = concatenate([branch1, branch2, branch31, branch32, branch41, branch42], axis=-1, name=f'{name}.merger')
    return x


def Inception_v4(
    filters,
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
    
    # Stem block
    x = stem_block(
        inputs=inputs,
        filters=filters,
        **layer_constant_dict,
        name="stem"
    )
    
    # Inception-A
    for i in range(num_blocks[0]):
        x = inception_A(
            inputs=x,
            filters=filters * 2,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = reduction_A(
        inputs=x,
        k=192,
        l=224,
        m=256,
        n=384,
        **layer_constant_dict,
        name=f"stage1.block{i + 2}"
    )

    # Inception-B
    for i in range(num_blocks[1]):
        x = inception_B(
            inputs=x,
            filters=filters * 4,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    # Reduction-B
    x = reduction_B(
        inputs=x,
        filters=filters * 6,
        **layer_constant_dict,
        name=f"stage2.block{i + 2}"
    )
                  
    # Inception-C
    for i in range(num_blocks[2]):
        x = inception_C(
            inputs=x,
            filters=filters * 8,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(8, 8), strides=(1, 1)),
            Dropout(rate=drop_rate),
            Flatten(),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "Inception-v4"
    if num_blocks == [4, 7, 3]:
        model_name += "-base"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Inception_v4_backbone(
    filters,
    num_blocks,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block3",
        "stem.merger2",
        f"stage1.block{num_blocks[0]}.merger",
        f"stage2.block{num_blocks[1]}.merger",
    ]
    
    return create_model_backbone(
        model_fn=Inception_v4,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_base_v4(
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
    
    model = Inception_v4(
        filters=32,
        num_blocks=[4, 7, 3],
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


def Inception_base_v4_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.block3",
        "stem.merger2",
        "stage1.block4.merger",
        "stage2.block7.merger",
    ]

    return create_model_backbone(
        model_fn=Inception_base_v4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
