"""
    VGG: Deep Convolutional Network with Simple and Uniform Architecture
    
    Overview:
        VGG (Visual Geometry Group) is a classic CNN backbone architecture known for its
        **uniform structure**, using only **3×3 convolutions** and **2×2 max-pooling** in a deep stack.
        Proposed by Oxford’s VGG team in 2014, it was a major milestone in demonstrating
        that **depth improves performance** significantly in image classification.
    
        Key innovations include:
            - Very deep yet simple design (up to 19 layers)
            - Uniform use of 3×3 Conv and 2×2 MaxPooling
            - High parameter count, but easy to implement and extend
    
    Key Components:
        • VGG Block:
            - Repeats multiple 3×3 Conv layers with ReLU
            - Followed by a 2×2 MaxPooling (stride=2) to downsample

        • Full Architecture:
            - Input image (224×224) is passed through 5 convolutional stages
            - Each stage increases the channel depth (64 → 128 → 256 → 512 → 512)
            - Followed by 3 fully-connected layers for classification

        • No Skip Connections:
            - VGG uses plain stacking of conv layers (no residual/shortcut paths)
    
        • Variants:
            - **VGG11**: fewer conv layers
            - **VGG13, VGG16, VGG19**: progressively deeper
            - All use ReLU and max pooling, no batch norm (original), but many modern variants add BN
    
        • Characteristics:
            - High memory & parameter count
            - Still used in applications requiring **feature extraction**
            - Backbone in early object detection models (e.g., Fast R-CNN, SSD)
    
    Deployment:
        - VGG is easy to port to different frameworks
        - Compatible with pretrained weights from many sources
        - Despite its age, VGG features are robust for transfer learning

    General Model Architecture:
         -----------------------------------------------------------------------
        | Stage         | Layer                       | Output Shape            |
        |---------------+-----------------------------+-------------------------|
        | Input         | input_layer                 | (None, 224, 224, 3)     |
        |---------------+-----------------------------+-------------------------|
        | Stage 1       | ConvolutionBlock (x2)       | (None, 224, 224, 64)    |
        |               | MaxPooling2D (3x3, s=2)     | (None, 112, 112, 64)    |
        |---------------+-----------------------------+-------------------------|
        | Stage 2       | ConvolutionBlock (x2)       | (None, 112, 112, 128)   |
        |               | MaxPooling2D (3x3, s=2)     | (None, 56, 56, 128)     |
        |---------------+-----------------------------+-------------------------|
        | Stage 2       | ConvolutionBlock (x2)       | (None, 56, 56, 256)     |
        |               | MaxPooling2D (3x3, s=2)     | (None, 28, 28, 256)     |
        |---------------+-----------------------------+-------------------------|
        | Stage 2       | ConvolutionBlock (x2)       | (None, 28, 28, 512)     |
        |               | MaxPooling2D (3x3, s=2)     | (None, 14, 14, 512)     |
        |---------------+-----------------------------+-------------------------|
        | Stage 2       | ConvolutionBlock (x2)       | (None, 14, 14, 512)     |
        |               | MaxPooling2D (3x3, s=2)     | (None, 7, 7, 512)       |
        |---------------+-----------------------------+-------------------------|
        | CLS Logics    | Flatten                     | (None, 25088)           |
        |               | fc1                         | (None, 4096)            |
        |               | fc1                         | (None, 4096)            |
        |               | fc3 (Logits)                | (None, 1000)            |
         -----------------------------------------------------------------------
         
    Model Parameter Comparison:
         ---------------------------------------
        |     Model Name      |    Params       |
        |---------------------------------------|
        |       VGG-11        |   132,863,336   |
        |---------------------|-----------------|
        |       VGG-13        |   133,047,848   |
        |---------------------|-----------------|
        |       VGG-16        |   138,357,544   |
        |---------------------|-----------------|
        |       VGG-19        |   143,667,240   |
         ---------------------------------------

    References:
        - Paper: “Very Deep Convolutional Networks for Large-Scale Image Recognition”  
          https://arxiv.org/abs/1409.1556
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py    

"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout, MaxPooling2D,
)

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone


def VGG(
    filters,
    num_blocks,
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

    x = inputs
    for i, num_block in enumerate(num_blocks):
        f = filters * 2**(i - 1) if i > len(num_blocks) - 2 else filters * 2**i

        for j in range(num_block):
            x = Sequential([
                Conv2D(
                    filters=f,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation),
            ], name=f"stage{i + 1}.block{j + 1}")(x)
            
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"stage{i + 1}.pool")(x)

    if include_head:
        x = Sequential([
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

    model_name = "VGG"
    if filters == 64:
        if num_blocks == [1, 1, 2, 2, 2]:
            model_name += "-11"
        elif num_blocks == [2, 2, 2, 2, 2]:
            model_name += "-13"
        elif num_blocks == [2, 2, 3, 3, 3]:
            model_name += "-16"
        elif num_blocks == [2, 2, 4, 4, 4]:
            model_name += "-19"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def VGG_backbone(
    filters,
    num_blocks,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
        "stage4.pool",
    ]

    return create_model_backbone(
        model_fn=VGG,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def VGG11(
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
    
    model = VGG(
        filters=64,
        num_blocks=[1, 1, 2, 2, 2],
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


def VGG11_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
        "stage4.pool",
    ]

    return create_model_backbone(
        model_fn=VGG11,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def VGG13(
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
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 2, 2, 2],
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


def VGG13_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
        "stage4.pool",
    ]

    return create_model_backbone(
        model_fn=VGG13,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def VGG16(
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
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 3, 3, 3],
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


def VGG16_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
        "stage4.pool",
    ]

    return create_model_backbone(
        model_fn=VGG16,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def VGG19(
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
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 4, 4, 4],
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


def VGG19_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
        "stage4.pool",
    ]

    return create_model_backbone(
        model_fn=VGG19,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    