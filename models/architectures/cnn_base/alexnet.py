"""
    Overview:
        AlexNet is one of the pioneering deep convolutional neural networks 
        that dramatically improved image classification performance on ImageNet 
        and sparked the deep learning revolution in computer vision.
    
        It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 
        with a top-5 test error of 15.3%, compared to 26.2% of the second-best.
    
    Key Characteristics:
        - 8 layers (5 convolutional + 3 fully connected)
        - Uses ReLU activation (instead of tanh or sigmoid)
        - Local response normalization (LRN)
        - Overlapping max-pooling
        - Dropout used in fully connected layers to reduce overfitting

    General Model Architecture:
         ---------------------------------------------------------------------------------
        | Stage                  | Layer                         | Output Shape           |
        |------------------------+-------------------------------+------------------------|
        | Input                  | input_layer                   | (None, 227, 227, 3)    |
        |------------------------+-------------------------------+------------------------|
        | Stem                   | ConvolutionBlock (11x11, s=4) | (None, 55, 55, 96)     |
        |                        | MaxPooling2D (3x3, s=2)       | (None, 27, 27, 96)     |
        |------------------------+-------------------------------+------------------------|
        | Stage 1                | ConvolutionBlock (5x5, p=2)   | (None, 27, 27, 256)    |
        |                        | MaxPooling2D (3x3, s=2)       | (None, 13, 13, 256)    |
        |------------------------+-------------------------------+------------------------|
        | Stage 2                | ConvolutionBlock (3x3, p=1)   | (None, 13, 13, 384)    |
        |                        | ConvolutionBlock (3x3, p=1)   | (None, 13, 13, 384)    |
        |                        | ConvolutionBlock (3x3, p=1)   | (None, 13, 13, 256)    |
        |                        | MaxPooling2D (3x3, s=2)       | (None, 6, 6, 256)      |
        |------------------------+-------------------------------+------------------------|
        | CLS Logics             | Flatten                       | (None, 9216)           |
        |                        | fc1                           | (None, 4096)           |
        |                        | fc2                           | (None, 4096)           |
        |                        | fc3 (Logits)                  | (None, 1000)           |
         ---------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         -------------------------------------
        |     Model Name    |    Params       |
        |-------------------+-----------------|
        |      AlexNet      |   50,844,008    |
         -------------------------------------

    References:
        - Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
          Krizhevsky, Sutskever, Hinton. NeurIPS 2012.
          https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
          
        - PyTorch repository:
          https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense,
    Dropout, MaxPooling2D,
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, create_model_backbone, check_regularizer



def AlexNet(
    inputs=[227, 227, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5
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
        default_size=227,
        min_size=32,
        weights=weights,
    )

    # Stage 0:
    x = Sequential([
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
        get_normalizer_from_name(
            "local-response-norm",
            depth_radius=5,
            alpha=0.0001,
            beta=0.75,
        ),
    ], name="stem.block1")(inputs)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)

    # Stage 1:
    x = Sequential([
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
        get_normalizer_from_name(
            "local-response-norm",
            depth_radius=5,
            alpha=0.0001,
            beta=0.75,
        ),
    ], name="stage1.block1")(x)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage1.pool")(x)

    # Stage 2:
    for i in range(2):
        x = Sequential([
            Conv2D(
                filters=384,
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

    
    x = Sequential([
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stage2.block3")(x)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage2.pool")(x)

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Flatten(),
            Dense(units=4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            Dense(units=4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model = Model(inputs=inputs, outputs=x, name="AlexNet")
    return model


def AlexNet_backbone(
    inputs=[227, 227, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem.norm",
        "stage1.norm",
        "stage3.activ",
    ]

    return create_model_backbone(
        model_fn=AlexNet,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    