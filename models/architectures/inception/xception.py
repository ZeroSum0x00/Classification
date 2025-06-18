"""
    Xception: Deep Separable Convolutional Backbone with Residual and Entry/Exit Flow Design
    
    Overview:
        Xception (Extreme Inception) is a convolutional neural network architecture
        that replaces traditional Inception modules with depthwise separable convolutions,
        combined with residual connections. This design leads to a highly efficient and
        scalable backbone that balances performance and computational cost.
    
        Xception can be seen as a linear stack of depthwise separable convolution layers
        with shortcut connections, providing superior performance over Inception-v3
        in classification and dense prediction tasks.
    
        Key innovations include:
            - Depthwise Separable Convolutions: Factorization of spatial and channel-wise filtering
            - Entry/Middle/Exit Flow: Structured pipeline with residual connections
            - Fully Convolutional and Efficient: Suitable for both classification and dense tasks
    
    Key Components:
        • Depthwise Separable Convolution:
            - Replaces a standard convolution with:
                1. **Depthwise Conv**: Applies 1 filter per input channel
                2. **Pointwise Conv (1×1)**: Combines outputs across channels
            - Greatly reduces parameters and FLOPs while maintaining accuracy.
    
        • Entry Flow:
            - Initial convolutional layers for downsampling and feature extraction.
            - Includes 2 conv layers followed by 3 residual blocks:
                • Each residual block: 1×1 shortcut + depthwise separable conv stack
                • Spatial size reduced via stride-2 depthwise convs
            - Output size reduced from input to ~35×35 (on 299×299 input)
    
        • Middle Flow:
            - Main feature extraction body:
                • 8 repeated blocks, each consisting of 3 separable conv layers + ReLU + residual connection
                • Resolution is preserved
            - Acts like a residual tower, deeply processes the features.
    
        • Exit Flow:
            - Final feature processing and downsampling:
                • Two separable conv layers with residual path
                • Followed by another separable conv block
            - Output resolution reduced to ~10×10 (for 299×299 input)
    
        • Global Average Pooling and Classifier:
            - GAP reduces spatial dimensions to vector
            - Final fully connected layer for classification (e.g., 1000 classes on ImageNet)

    General Model Architecture:
         ----------------------------------------------------------------------------------
        | Stage                  | Layer                         | Output Shape            |
        |------------------------+-------------------------------+-------------------------|
        | Input                  | input_layer                   | (None, 299, 299, 3)     |
        |------------------------+-------------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=2)   | (None, 149, 149, 32)    |
        |                        | ConvolutionBlock (3x3, s=1)   | (None, 147, 147, 64)    |
        |------------------------+-------------------------------+-------------------------|
        | Stage 1                | xception_block                | (None, 74, 74, 128)     |
        |------------------------+-------------------------------+-------------------------|
        | Stage 2                | xception_block                | (None, 37, 37, 256)     |
        |------------------------+-------------------------------+-------------------------|
        | Stage 3                | xception_block                | (None, 19, 19, 736)     |
        |------------------------+-------------------------------+-------------------------|
        | Stage 4                | xception_separable_block (8x) | (None, 19, 19, 736)     |
        |------------------------+-------------------------------+-------------------------|
        | Stage 5                | xception_block                | (None, 10, 10, 1024)    |
        |------------------------+-------------------------------+-------------------------|
        | Stage 6                | xception_block                | (None, 10, 10, 2048)    |
        |------------------------+-------------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling2D        | (None, 2048)            |
        |                        | fc (Logics)                   | (None, 1000)            |
         ----------------------------------------------------------------------------------

    Model Parameter Comparison:
         --------------------------------------
        |     Model Name      |    Params      |
        |--------------------------------------|
        |     Xception        |   23,238,312   |
         --------------------------------------

    References:
        - Paper: “Xception: Deep Learning with Depthwise Separable Convolutions”  
          https://arxiv.org/abs/1610.02357
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
    
        - PyTorch implementation:  
          https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, Dense,
    Dropout, MaxPooling2D, GlobalAveragePooling2D,
    add,
)

from .inception_v3 import convolution_block
from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone



def xception_block(
    inputs,
    filters,
    shortcut=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"xception_block_{K.get_uid('xception_block')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    
    x = Sequential([
        get_activation_from_name(activation),
        SeparableConv2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=regularizer_decay,
            pointwise_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        SeparableConv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=regularizer_decay,
            pointwise_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.separable")(inputs)

    if shortcut:
        residual = Sequential([
            Conv2D(
                filters=filters[1],
                kernel_size=(1, 1),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"{name}.residual")(inputs)
    
        x = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
            name=f"{name}.pooling"
        )(x)
    
        x = add([x, residual], name=f"{name}.add")

    x = LinearLayer(name=f"{name}.linear")(x)
    return x


def Xception(
    filters,
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
        min_size=71,
        weights=weights
    )
    
    # Block 1
    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * 2,
            kernel_size=(3, 3),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name="stem")(inputs)

    x = xception_block(x, filters=[filters * 4, filters * 4], name="stage1")
    x = xception_block(x, filters=[filters * 8, filters * 8], name="stage2")
    x = xception_block(x, filters=[filters * 23, filters * 23], name="stage3")

    for i in range(8):
        residual = x
        x = Sequential([
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=regularizer_decay,
                pointwise_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=regularizer_decay,
                pointwise_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=regularizer_decay,
                pointwise_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"stage4.block{i + 1}.separable")(x)
        
        x = add([x, residual], name=f"stage4.block{i + 1}.linear")

    x = xception_block(x, filters=[filters * 23, filters * 32], name="stage5")
    x = xception_block(x, filters=[filters * 48, filters * 64], shortcut=False, name="stage6")

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "Xception"
    if filters == 32:
        model_name += "-base"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Xception_backbone(
    filters,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.linear",
        "stage2.linear",
        "stage4.block8.linear",
    ]

    return create_model_backbone(
        model_fn=Xception,
        custom_layers=custom_layers,
        filters=filters,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Xception_base(
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
    
    model = Xception(
        filters=32,
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

def Xception_base_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.linear",
        "stage2.linear",
        "stage4.block8.linear",
    ]

    return create_model_backbone(
        model_fn=Xception_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
