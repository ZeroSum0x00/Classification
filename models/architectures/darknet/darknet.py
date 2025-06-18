"""
    Overview:
        The YOLOv1 backbone is a custom convolutional neural network architecture 
        introduced in the paper "You Only Look Once: Unified, Real-Time Object Detection".
        It was designed for real-time object detection by learning to extract rich features 
        directly from raw image pixels in a single network pass.

        YOLOv1 uses a CNN with 24 convolutional layers followed by 2 fully connected layers.
        The model is trained to predict bounding boxes and class probabilities simultaneously.

    Key Characteristics:
        - 24 convolutional layers + 2 fully connected layers
        - Uses LeakyReLU activation (α = 0.1) throughout
        - Large initial convolution kernel (7x7) with stride 2
        - Several 1x1 convolution layers for dimensionality reduction
        - Max-pooling layers with overlapping strides
        - Dense feature representation with high spatial resolution
        - Fully connected layers used for prediction

    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 416, 416, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=1) | (None, 416, 416, C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 208, 208, 2C)    |
        |                        | CSPDarkNetBlock (1x)        | (None, 208, 208, 2C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | ConvolutionBlock (3x3, s=2) | (None, 104, 104, 4C)    |
        |                        | CSPDarkNetBlock (2x)        | (None, 104, 104, 4C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | ConvolutionBlock (3x3, s=2) | (None, 52, 52, 8C)      |
        |                        | CSPDarkNetBlock (8x)        | (None, 52, 52, 8C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | ConvolutionBlock (3x3, s=2) | (None, 26, 26, 16C)     |
        |                        | CSPDarkNetBlock (8x)        | (None, 26, 26, 16C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 5                | ConvolutionBlock (3x3, s=2) | (None, 13, 13, 32C*S)   |
        |                        | CSPDarkNetBlock (4x)        | (None, 13, 13, 32C*S)   |
        |                        | pyramid_poolings (*)        | (None, 13, 13, 32C*S)   |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 32C*S)           |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------
        (*) Note: While the original architecture does not include a Pyramid Pooling layer, 
        it can be optionally incorporated to enhance feature aggregation and create an extended variant of the model.

    Model Parameter Comparison:
         -----------------------------------------
        |      Model Name      |     Params       |
        |----------------------+------------------|
        |      DarkNet19       |   265,680,960    |
         -----------------------------------------

    References:
        - Paper: "You Only Look Once: Unified, Real-Time Object Detection"
          https://arxiv.org/pdf/1506.02640.pdf
          
        - Original implementation:
          https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Dense,
    Dropout, MaxPooling2D, GlobalAveragePooling2D,
    concatenate, Activation, Flatten, Dropout
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
)
from utils.model_processing import (
    process_model_input, create_model_backbone, create_layer_instance,
    check_regularizer, validate_conv_arg,
)



def DarkNet(
    filters=64,
    num_blocks=[1, 2, 3, 6, 3, 3],
    channel_scale=2,
    final_channel_scale=1,
    inputs=[416, 416, 3],
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')
        
    regularizer_decay = check_regularizer(regularizer_decay)
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=416,
        min_size=32,
        weights=weights
    )

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks) - 1)]
    
    x = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)

    for i in range(num_blocks[1]):
        if i == 0:
            x = MaxPooling2D(2, 2, name=f"stage1.block{i + 1}")(x)
        else:
            x = Sequential([
                Conv2D(
                    filters=filters[0] * 3,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage1.block{i + 1}")(x)

    for i in range(num_blocks[2]):
        if i == 0:
            x = MaxPooling2D(2, 2, name=f"stage2.block{i + 1}")(x)
        else:
            x = Sequential([
                Conv2D(
                    filters=filters[i],
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
                Conv2D(
                    filters=filters[i + 1],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage2.block{i + 1}")(x)

    for i in range(num_blocks[3]):
        if i == 0:
            x = MaxPooling2D(2, 2, name=f"stage3.block{i + 1}")(x)
        else:
            f = filters[3] if i == num_blocks[3] - 1 else filters[2]
            
            x = Sequential([
                Conv2D(
                    filters=f,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                Conv2D(
                    filters=f * 2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage3.block{i + 1}")(x)

    for i in range(num_blocks[4]):
        if i == 0:
            x = MaxPooling2D(2, 2, name=f"stage4.block{i + 1}")(x)
        else:
            f = filters[3]
            
            x = Sequential([
                Conv2D(
                    filters=f,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                Conv2D(
                    filters=f * 2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage4.block{i + 1}")(x)

    for i in range(num_blocks[5]):
        if i == 0:
            x = Sequential([
                Conv2D(
                    filters=filters[4],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
                Conv2D(
                    filters=filters[4],
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage5.block{i + 1}")(x)
        else:
            x = Sequential([
                Conv2D(
                    filters=filters[4],
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage5.block{i + 1}")(x)

    if include_head:
        x = Sequential([
            Flatten(),
            Dense(units=4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    
    model_name = "DarkNet"
    if filters == [32, 64, 128, 256, 512, 1024] and num_blocks == [1, 2, 4, 4, 6, 6]:
        model_name += "-base"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNet_base(
    inputs=[416, 416, 3],
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNet(
        filters=64,
        num_blocks=[1, 2, 3, 6, 3, 3],
        channel_scale=2,
        final_channel_scale=1,
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


def DarkNet_base_backbone(
    inputs=[416, 416, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block3",
        "stage3.block6",
        "stage4.block3",
    ]

    return create_model_backbone(
        model_fn=DarkNet_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    