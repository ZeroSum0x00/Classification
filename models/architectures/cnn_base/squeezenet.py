"""
    # Overview:
        SqueezeNet is a lightweight convolutional neural network architecture
        designed to achieve AlexNet-level accuracy on ImageNet with significantly
        fewer parameters — making it ideal for deployment on mobile and embedded systems.
    
    # Key Features:
        - Achieves AlexNet-level accuracy with **~50× fewer parameters** (~1.2M vs ~60M).
        - Utilizes **Fire Modules**:
            * A "squeeze" layer (1x1 conv) that reduces depth.
            * Followed by an "expand" layer with a mix of 1x1 and 3x3 convolutions.
        - No fully connected layers — replaced with convolution + global average pooling.
        - Small model size (~5MB), ideal for low-latency or memory-constrained environments.

    General Model Architecture:
         -------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape           |
        |------------------------+-----------------------------+------------------------|
        | Input                  | input_layer                 | (None, 224, 224, 3)    |
        |------------------------+-----------------------------+------------------------|
        | Stem                   | ConvolutionBlock (7x7, s=2) | (None, 111, 111, 96)   |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 55, 55, 96)     |
        |------------------------+-----------------------------+------------------------|
        | Stage 1                | fire_module1                | (None, 55, 55, 128)    |
        |                        | fire_module2                | (None, 55, 55, 128)    |
        |                        | fire_module3                | (None, 55, 55, 256)    |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 27, 27, 256)    |
        |------------------------+-----------------------------+------------------------|
        | Stage 2                | fire_module4                | (None, 27, 27, 256)    |
        |                        | fire_module5                | (None, 27, 27, 384)    |
        |                        | fire_module6                | (None, 27, 27, 384)    |
        |                        | fire_module7                | (None, 27, 27, 512)    |
        |                        | MaxPooling2D (3x3, s=2)     | (None, 13, 13, 512)    |
        |------------------------+-----------------------------+------------------------|
        | Stage 3                | fire_module8                | (None, 13, 13, 512)    |
        |                        | ConvolutionBlock (1x1, s=1) | (None, 13, 13, 1000)   |
        |------------------------+-----------------------------+------------------------|
        | CLS Logics             | AveragePooling              | (None, 1, 1, 1000)     |
        |                        | Flatten                     | (None, 1000)           |
        |                        | fc (Logics)                 | (None, 1000)           |
         -------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         -------------------------------------
        |     Model Name    |    Params       |
        |-------------------------------------|
        |    SqueezeNet     |    2,237,904    |
         -------------------------------------

    # Reference:
        - Paper: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
          https://arxiv.org/abs/1602.07360
          
        - PyTorch repository:
          https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/SqueezeNet.py
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dropout, Dense,
    MaxPooling2D, AveragePooling2D,
    concatenate,
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, create_model_backbone, check_regularizer



def fire_module(
    inputs,
    filters,
    squeeze_channel,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"fire_module_{K.get_uid('fire_module')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        Conv2D(
            filters=squeeze_channel,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.conv_block")(inputs)

    expand_1x1 = Sequential([
        Conv2D(
            filters=filters // 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.expand_1x1")(x)
    
    expand_3x3 = Sequential([
        Conv2D(
            filters=filters // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.expand_3x3")(x)
    
    return concatenate([expand_1x1, expand_3x3], name=f"{name}.fusion")


def SqueezeNet(
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
        weights=weights,
    )

    # Stage 0:
    x = Sequential([
        Conv2D(
            filters=96,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem.block1")(inputs)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)

    # Stage 1:
    for i in range(2):
        x = fire_module(
            inputs=x,
            filters=128,
            squeeze_channel=16,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = fire_module(
        inputs=x,
        filters=256,
        squeeze_channel=32,
        **layer_constant_dict,
        name="stage1.block3"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage1.pool")(x)

    # Stage 2:
    x = fire_module(
        inputs=x,
        filters=256,
        squeeze_channel=32,
        **layer_constant_dict,
        name="stage2.block1"
    )
    
    for i in range(2):
        x = fire_module(
            inputs=x,
            filters=384,
            squeeze_channel=48,
            **layer_constant_dict,
            name=f"stage2.block{i + 2}"
        )
        
    x = fire_module(
        inputs=x,
        filters=512,
        squeeze_channel=64,
        **layer_constant_dict,
        name="stage2.block4"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage2.pool")(x)

    # Stage 3:
    x = fire_module(
        inputs=x,
        filters=512,
        squeeze_channel=64,
        **layer_constant_dict,
        name="stage3.block1"
    )
    
    x = Sequential([
        Dropout(rate=drop_rate),
        Conv2D(
            filters=1 if num_classes == 2 else num_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
    ], name="stage3.block2")(x)

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(13, 13), strides=(1, 1)),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model = Model(inputs=inputs, outputs=x, name="SqueezeNet")
    return model


def SqueezeNet_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block3.fusion",
        "stage2.block4.fusion",
    ]

    return create_model_backbone(
        model_fn=SqueezeNet,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    