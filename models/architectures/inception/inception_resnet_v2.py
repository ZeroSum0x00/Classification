"""
    Inception ResNet V2: Deep Hybrid Backbone with Residual Inception Blocks and Scalable Architecture
    
    Overview:
        Inception-ResNet-v2 is a powerful convolutional backbone that combines
        the rich multi-scale feature extraction of Inception modules with the
        training speed and stability of residual connections.
    
        It improves upon Inception-ResNet-v1 by increasing depth, regularizing structure,
        and refining block designs (e.g., more factorized convolutions and deeper residual flows).
        The architecture is highly optimized for both classification and transfer learning tasks.
    
        Key innovations include:
            - Scaled Residual Inception Blocks: Deeper and wider than v1 for better performance
            - Structured Grid Reduction: Efficient spatial downsampling while preserving context
            - BatchNorm and Residual Scaling: Improve training stability and generalization
    
    Key Components:
        • Stem Block:
            - The initial feature extractor, using multiple conv and pooling paths.
            - Replaces large kernel filters (e.g., 7×7) with factorized sequences (3×3).
            - Outputs a rich, high-resolution feature map.
    
        • Inception-ResNet Blocks:
            - All blocks follow:
              **Multi-branch convolution → Concatenation → 1×1 Conv → Scaled Residual Add**
            - Residual path is scaled by a small factor (e.g., 0.1–0.2) to prevent gradient explosion.
    
            Block types:
            - **Block35 (Inception-ResNet-A)**:  
              • Used at 35×35 resolution  
              • Uses 1×1, 3×3, and 3×3 stacked convs  
              • Repeated ~10×
    
            - **Block17 (Inception-ResNet-B)**:  
              • Used at 17×17 resolution  
              • Includes asymmetric 1×7 + 7×1 convolutions  
              • Repeated ~20×
    
            - **Block8 (Inception-ResNet-C)**:  
              • Used at 8×8 resolution  
              • Wider 1×1 and 3×3 branches  
              • Repeated ~10×
    
        • Reduction Blocks:
            - Designed to efficiently reduce spatial size between block stages:
                - **Reduction-A**: 35×35 → 17×17
                - **Reduction-B**: 17×17 → 8×8
            - Multi-branch structure: uses stride-2 convs, stacked convolutions, and max pooling.
    
        • Global Average Pooling:
            - Reduces feature maps to a vector before classification.
            - Typically followed by dropout and a fully connected softmax layer.

        • Residual Scaling:
            - Each block adds its output with the input multiplied by a small scalar (e.g., 0.1),
              which stabilizes training even with very deep models.

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
        | Stage 1                | inception_resnet_A (10x)    | (None, 35, 35, 384)     |
        |                        | reduction_A                 | (None, 17, 17, 1152)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | inception_resnet_B (20x)    | (None, 17, 17, 1152)    |
        |                        | reduction_B                 | (None, 8, 8, 2144)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | inception_resnet_C (5x)     | (None, 8, 8, 2144)      |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | AveragePooling2D (8x8, s=1) | (None, 1, 1, 2144)      |
        |                        | Flatten                     | (None, 2144)            |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         ---------------------------------------------
        |         Model Name        |     Params      |
        |---------------------------------------------|
        |    Inception Resnet v2    |    55,463,816   |
         ---------------------------------------------

    References:
        - Paper: “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning”  
          https://arxiv.org/abs/1602.07261
    
        - TensorFlow/Keras implementation:
          https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v2.py
    
        - PyTorch implementation:  
          https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py
          
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Lambda, Flatten, Dense,
    Dropout, MaxPooling2D, AveragePooling2D,
    concatenate, add,
)

from .inception_resnet_v1 import convolution_block, reduction_A
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone



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
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv3"
    )
    
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.step1.branch1.pool"
    )(x)
    
    branch2 = convolution_block(
        inputs=x,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step1.branch2.conv"
    )
    
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger1")
    
    branch1 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch1.conv1"
    )
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch1.conv2"
    )
    
    branch2 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(7, 1),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(1, 7),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch2.conv3"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step2.branch2.conv4"
    )

    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger2")

    branch1 = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.step3.branch1.conv"
    )
    
    branch2 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.step3.branch2.conv"
    )(x)
    
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger3")
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
        name=f"{name}.branch2.conv1"
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
        name=f"{name}.branch2.conv2"
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
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.conv2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters * 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.conv3"
    )

    # merger
    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 12,
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
        filters=int(filters * 3/2),
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
        branch2,
        filters=int(filters * 5/4),
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
        filters=int(filters * 3/2),
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step3"
    )

    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 9,
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
        filters=int(filters * 9/8),
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
        filters=int(filters * 9/8),
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
        filters=int(filters * 5/4),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step3"
    )
    
    out = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def inception_resnet_C(
    inputs,
    filters=192,
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
        filters=int(filters * 7/6),
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
        filters=int(filters * 4/3),
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
        filters=int(filters * 67/6),
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


def Inception_Resnet_v2(
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
        k=256,
        l=256,
        m=384,
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
            filters=filters * 6,
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

    model_name = "Inception-Resnet-v2"
    if filters == 32 and num_blocks == [1, 2, 1] and scale_residual:
        model_name = "-base"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Inception_Resnet_v2_backbone(
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
        "stem.conv3.conv_block",
        "stem.merger2",
        f"stage1.block{num_blocks[0]}.activ",
        f"stage2.block{num_blocks[1]}.activ",
    ]
    
    return create_model_backbone(
        model_fn=Inception_Resnet_v2,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        scale_residual=scale_residual,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Inception_Resnet_base_v2(
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
    
    model = Inception_Resnet_v2(
        filters=32,
        num_blocks=[10, 20, 10],
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


def Inception_Resnet_base_v2_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.conv3.conv_block",
        "stem.merger2",
        "stage1.block10.activ",
        "stage2.block20.activ",
    ]

    return create_model_backbone(
        model_fn=Inception_Resnet_base_v2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
