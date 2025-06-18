"""
    ResNet: Deep Residual Backbone with Identity Shortcut Connections
    
    Overview:
        ResNet (Residual Network) is a foundational CNN architecture that introduced
        **residual connections**, enabling the training of very deep networks by
        addressing the vanishing gradient problem. Its design has made it a standard
        backbone for a wide range of vision tasks (classification, detection, segmentation).
    
        Key innovations include:
            - Identity Shortcut Connections: Bypass path for better gradient flow
            - Residual Learning: Models the residual function F(x) = H(x) - x
            - Deep but Efficient: Enables networks with 50, 101, 152+ layers
    
    Key Components:
        • Residual Block:
            - Main unit of ResNet architecture
            - Composed of 2 or 3 convolutions + BatchNorm + ReLU, with a shortcut connection:

            - If input and output dimensions differ, the shortcut has a 1×1 conv (projection)
            - Two types:
                - **BasicBlock** (used in ResNet-18/34)
                - **Bottleneck** (used in ResNet-50/101/152): 1x1 → 3x3 → 1x1 conv
    
        • Bottleneck Block:
            - 1×1 conv (reduce channels) →
            - 3×3 conv (process features) →
            - 1×1 conv (expand channels) →
            - Residual add + ReLU
    
        • Downsampling:
            - Achieved by:
                - Stride = 2 in some convolutions (usually 1st conv of a stage)
                - 1×1 conv in shortcut if dimensions change
    
        • Network Staging:
            - The network is structured in multiple stages:

        • Variants:
            - **ResNet-18 / 34**: Use BasicBlock
            - **ResNet-50 / 101 / 152**: Use BottleneckBlock
            - Pretrained variants widely available

    General Model Architecture:
         -------------------------------------------------------------------------
        | Stage         | Layer                         | Output Shape            |
        |---------------+-------------------------------+-------------------------|
        | Input         | input_layer                   | (None, 224, 224, 3)     |
        |---------------+-------------------------------+-------------------------|
        | Stem          | ZeroPadding2D (3x3)           | (None, 230, 230, 3)     |
        |               | ConvolutionBlock (7x7, s=2)   | (None, 112, 112, 64)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 1       | MaxPooling2D (3x3, s=2)       | (None, 55, 55, 64)      |
        |               | resnet_block (x3)             | (None, 55, 55, 256)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 2       | resnet_block (s=2)            | (None, 28, 28, 512)     |
        |               | resnet_block (x3)             | (None, 28, 28, 512)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 3       | resnet_block (s=2)            | (None, 14, 14, 1024)    |
        |               | resnet_block (x5)             | (None, 14, 14, 1024)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 4       | resnet_block (s=2)            | (None, 7, 7, 2048)      |
        |               | resnet_block (x2)             | (None, 7, 7, 2048)      |
        |---------------+-------------------------------+-------------------------|
        | CLS Logics    | AveragePooling2D              | (None, 1, 1, 2048)      |
        |               | Flatten                       | (None, 2048)            |
        |               | fc (Logics)                   | (None, 1000)            |
         -------------------------------------------------------------------------

    Model Parameter Comparison:
         --------------------------------------
        |     Model Name      |    Params      |
        |--------------------------------------|
        |     ResNet-18       |   11,708,328   |
        |---------------------|----------------|
        |     ResNet-34       |   21,827,624   |
        |---------------------|----------------|
        |     ResNet-50       |   25,636,712   |
        |---------------------|----------------|
        |     ResNet-101      |   44,707,176   |
        |---------------------|----------------|
        |     ResNet-152      |   60,419,944   |
         --------------------------------------

    References:
        - Paper: “Deep Residual Learning for Image Recognition”  
          https://arxiv.org/abs/1512.03385
    
        - Official PyTorch repository:  
          https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    
        - TensorFlow/Keras implementation:
          https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Flatten,
    MaxPooling2D, AveragePooling2D,
    Dense, Dropout, GlobalAveragePooling2D,
    add
)

from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, correct_pad,
    validate_conv_arg, check_regularizer,
    create_model_backbone,
)



def basic_block(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    use_bias=True,
    residual=False,
    use_final_activ=False,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"basic_block_{K.get_uid('basic_block')}"

    shortcut = inputs
    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)
    
    x = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.conv_block")(inputs)

    if residual:
        shortcut = Sequential([
            Conv2D(
                filters=filters[1],
                kernel_size=(1, 1),
                strides=strides,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"{name}.shortcut")(shortcut)

    x = add([x, shortcut], name=f"{name}.fusion")

    if use_final_activ:
        x = get_activation_from_name(activation, name=f"{name}.final_activ")(x)

    return x


def bottle_neck(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    use_bias=True,
    residual=False,
    use_final_activ=False,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"bottle_neck_block_{K.get_uid('bottleneck')}"
        
    shortcut = inputs
    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)
    
    x = Sequential([
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
        Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.conv_block")(inputs)
    
    if residual:
        shortcut = Sequential([
            Conv2D(
                filters=filters[2],
                kernel_size=(1, 1),
                strides=strides,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"{name}.shortcut")(shortcut)

    x = add([x, shortcut], name=f"{name}.fusion")

    if use_final_activ:
        x = get_activation_from_name(activation, name=f"{name}.final_activ")(x)

    return x


def ResNet(
    filters,
    block,
    num_blocks,
    inputs=[224, 224, 3],
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

    def get_filters(filters):
        return [filters, filters, filters * 4] if block == bottle_neck else [filters, filters]
        
    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')
        
    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "use_bias": True,
        "use_final_activ": True,
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
        weights=weights
    )

    x = Sequential([
        ZeroPadding2D(padding=(3, 3)),
        Conv2D(
            filters=filters,
            kernel_size=(7, 7),
            strides=(2, 2),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)

    for i, num_block in enumerate(num_blocks):
        f = get_filters(filters * 2**i)
        residual = True
        for j in range(num_block):
            if i == 0 and j == 0:
                x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f"stage{i + 1}.block{j + 1}")(x)
            else:
                x = block(
                    inputs=x,
                    filters=f,
                    kernel_size=(3, 3),
                    strides=(2, 2) if (i != 0 and j == 0) else (1, 1),
                    residual=residual,
                    **layer_constant_dict,
                    name=f"stage{i + 1}.block{j + 1}"
                )
                residual = False

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(7, 7)),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "ResNet"
    if block == basic_block:
        if num_blocks == [3, 2, 2, 2]:
            model_name += "-18"
        elif num_blocks == [4, 4, 6, 3]:
            model_name += "-34"
    elif block == bottle_neck:
        if num_blocks == [4, 4, 6, 3]:
            model_name += "-50"
        elif num_blocks == [4, 4, 23, 3]:
            model_name += "-101"
        elif num_blocks == [4, 8, 36, 3]:
            model_name += "-152"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ResNet_backbone(
    filters,
    block,
    num_blocks,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet,
        custom_layers=custom_layers,
        filters=filters,
        block=block,
        num_blocks=num_blocks,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ResNet18(
    inputs=[224, 224, 3],
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
    
    model = ResNet(
        filters=64,
        block=basic_block,
        num_blocks=[3, 2, 2, 2],
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


def ResNet18_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet18,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ResNet34(
    inputs=[224, 224, 3],
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
    
    model = ResNet(
        filters=64,
        block=basic_block,
        num_blocks=[4, 4, 6, 3],
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


def ResNet34_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet34,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ResNet50(
    inputs=[224, 224, 3],
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
    
    model = ResNet(
        filters=64,
        block=bottle_neck,
        num_blocks=[4, 4, 6, 3],
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


def ResNet50_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet50,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ResNet101(
    inputs=[224, 224, 3],
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
    
    model = ResNet(
        filters=64,
        block=bottle_neck,
        num_blocks=[4, 4, 23, 3],
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


def ResNet101_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet101,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ResNet152(
    inputs=[224, 224, 3],
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
    
    model = ResNet(
        filters=64,
        block=bottle_neck,
        num_blocks=[4, 8, 36, 3],
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


def ResNet152_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    return create_model_backbone(
        model_fn=ResNet152,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    