"""
    Res2Net: Multi-Scale Backbone with Scaled Residual Feature Splitting
    
    Overview:
        Res2Net is a CNN backbone that introduces **multi-scale feature extraction
        within a single residual block**, significantly enhancing the network’s ability
        to represent features at different granularities. It modifies the traditional
        residual block to perform hierarchical feature fusion across multiple channel groups.
    
        Key innovations include:
            - Scaled Residual Blocks: Replace single 3×3 conv with multi-branch 3×3 convs
            - Hierarchical Feature Fusion: Inter-group residual flow enhances scale representation
            - Compatible with ResNet, SE, and other CNN backbones
    
    Key Components:
        • Res2Net Block (Split → Transform → Fuse):
            - Input feature `X` is split into `s` channel groups: `[x₁, x₂, ..., xₛ]`
            - Grouped transformation:
                - x₁ → identity
                - x₂ → Conv3×3(x₂)
                - x₃ → Conv3×3(x₃ + output of x₂)
                - ...
                - xₛ → Conv3×3(xₛ + output of xₛ₋₁)
            - All transformed outputs are **concatenated** and passed to a 1×1 conv for fusion

        • Scale Parameter (s):
            - Defines the number of splits and thus the multi-scale granularity
            - Larger `s` gives stronger multi-scale features (e.g., s = 4, 6)
    
        • Full Res2Net Bottleneck Block:
            - Similar to ResNet Bottleneck but with:
                - Conv1x1 (Reduce channels) →
                - **Res2Net 3x3 module (Split + Fuse)** →
                - Conv1x1 (Restore channels) →
                - Add residual
    
        • Hierarchical Residual-Like Structure:
            - Promotes richer representation across varying receptive fields
            - Enhances both local and global context extraction within a block
    
        • Compatible with Downsampling:
            - Stride can be applied at the **first 3×3 conv** of each group
            - Keeps spatial resolution reduction clean and efficient

    General Model Architecture:
         -------------------------------------------------------------------------
        | Stage         | Layer                         | Output Shape            |
        |---------------+-------------------------------+-------------------------|
        | Input         | input_layer                   | (None, 224, 224, 3)     |
        |---------------+-------------------------------+-------------------------|
        | Stem          | ConvolutionBlock (7x7, s=2)   | (None, 112, 112, 64)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 1       | MaxPooling2D (3x3, s=2)       | (None, 55, 55, 64)      |
        |               | bottle_2_neck (x3)            | (None, 55, 55, 256)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 2       | bottle_2_neck (s=2)           | (None, 28, 28, 512)     |
        |               | bottle_2_neck (x3)            | (None, 28, 28, 512)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 3       | bottle_2_neck (s=2)           | (None, 14, 14, 1024)    |
        |               | bottle_2_neck (x5)            | (None, 14, 14, 1024)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 4       | bottle_2_neck (s=2)           | (None, 7, 7, 2048)      |
        |               | bottle_2_neck (x2)            | (None, 7, 7, 2048)      |
        |---------------+-------------------------------+-------------------------|
        | CLS Logics    | AveragePooling2D              | (None, 1, 1, 2048)      |
        |               | Flatten                       | (None, 2048)            |
        |               | fc (Logics)                   | (None, 1000)            |
         -------------------------------------------------------------------------

    Model Parameter Comparison:
         -------------------------------------------
        |        Model Name       |     Params      |
        |-------------------------------------------|
        |     Res2Net50           |    25,758,612   |
        |-------------------------------------------|
        |     Res2Net50-26w4s     |    25,758,612   |
        |-------------------------------------------|
        |     Res2Net50-26w6s     |    37,123,212   |
        |-------------------------------------------|
        |     Res2Net50-26w8s     |    48,487,812   |
        |-------------------------------------------|
        |     Res2Net50-48w2s     |    25,342,312   |
        |-------------------------------------------|
        |     Res2Net50-14w8s     |    25,122,612   |
        |-------------------------------------------|
        |     Res2Net101          |    45,325,748   |
        |-------------------------------------------|
        |     Res2Net101-26w4s    |    45,325,748   |
        |-------------------------------------------|
        |     Res2Net101-26w6s    |    67,270,060   |
        |-------------------------------------------|
        |     Res2Net101-26w8s    |    89,214,372   |
        |-------------------------------------------|
        |     Res2Net101-48w2s    |    44,460,648   |
        |-------------------------------------------|
        |     Res2Net101-14w8s    |    44,205,588   |
        |-------------------------------------------|
        |     Res2Net152          |    61,446,868   |
        |-------------------------------------------|
        |     Res2Net152-26w4s    |    61,446,868   |
        |-------------------------------------------|
        |     Res2Net152-26w6s    |    92,105,548   |
        |-------------------------------------------|
        |     Res2Net152-26w8s    |   122,764,228   |
        |-------------------------------------------|
        |     Res2Net152-48w2s    |    60,211,560   |
        |-------------------------------------------|
        |     Res2Net152-14w8s    |    59,928,436   |
         -------------------------------------------
    
    References:
        - Paper: “Res2Net: A New Multi-scale Backbone Architecture”  
          https://arxiv.org/abs/1904.01169
    
        - Pytorch implementation:
          https://github.com/Res2Net/Res2Net-PretrainedModels

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten,
    AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D,
    add, concatenate,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    AdaptiveAvgPooling2D, SplitWrapper,
)
from utils.model_processing import (
    process_model_input, validate_conv_arg,
    check_regularizer, create_model_backbone,
)



def bottle_2_neck(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    use_bias=True,
    residual=False,
    width_ratio=26,
    scale_ratio=4,
    activation="relu",
    normalizer="group-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    expansion = 4
    width = int(np.floor(filters * (width_ratio / 64.0)))
    nums = 1 if scale_ratio == 1 else scale_ratio - 1
    kernel_size = validate_conv_arg(kernel_size)
    strides = validate_conv_arg(strides)
    regularizer_decay = check_regularizer(regularizer_decay)

    shortcut = inputs
    
    if residual:
        shortcut = Sequential([
            AveragePooling2D(pool_size=strides, strides=strides, padding="same"),
            Conv2D(
                filters=filters * expansion,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"{name}.shortcut")(shortcut)

    x = Sequential([
        Conv2D(
            filters=width * scale_ratio,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.conv_block")(inputs)
    
    spx = SplitWrapper(num_or_size_splits=scale_ratio, axis=-1, name=f"{name}.split")(x)
    
    for i in range(nums):
        if i == 0 or residual:
            sp = spx[i]
        else:
            sp = sp + spx[i]

        sp = Sequential([
            Conv2D(
                filters=width,
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
        ], name=f"{name}.sp_block_{i + 1}")(sp)
        
        if i == 0:
            x = sp
        else:
            x = concatenate([x, sp], axis=-1, name=f"{name}.sp_concat_{i + 1}")

    if scale_ratio != 1:
        if residual:
            pool = AveragePooling2D(
                pool_size=(3, 3),
                strides=strides,
                padding='same',
                name=f'{name}.scale.pool'
            )(spx[nums])
            
            x = concatenate([x, pool], axis=-1, name=f"{name}.scale_concat_{i + 1}")
        else:
            x = concatenate([x, spx[nums]], axis=-1, name=f"{name}.scale_concat_{i + 1}")

    x = Sequential([
        Conv2D(
            filters=filters * expansion,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.post_conv_block")(x)
        
    x = add([x, shortcut], name=f"{name}.add")
    x = get_activation_from_name(activation, name=f'{name}.post_activ')(x)
    return x


def Res2Net(
    filters,
    num_blocks,
    width_ratio,
    scale_ratio,
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
        "use_bias": False,
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
        Conv2D(
            filters=filters,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)

    for i, num_block in enumerate(num_blocks):
        f = filters * 2**i
        residual = True
        for j in range(num_block):
            if i == 0 and j == 0:
                x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f"stage{i + 1}.block{j + 1}")(x)
            else:
                x = bottle_2_neck(
                    inputs=x,
                    filters=f,
                    kernel_size=(3, 3),
                    strides=(2, 2) if (i != 0 and j == 0) else (1, 1),
                    residual=residual,
                    width_ratio=width_ratio,
                    scale_ratio=scale_ratio,
                    **layer_constant_dict,
                    name=f"stage{i + 1}.block{j + 1}"
                )
                residual = False

    if include_head:
        x = Sequential([
            AdaptiveAvgPooling2D(output_size=1),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "Res2Net"
    if filters == 64 and num_blocks == [4, 4, 6, 3]:
        model_name += "-50"
    elif filters == 64 and num_blocks == [4, 4, 23, 3]:
        model_name += "-101"
    elif filters == 64 and num_blocks == [4, 8, 36, 3]:
        model_name += "-152"
    model_name += f"-{width_ratio}w"
    model_name += f"-{scale_ratio}s"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def Res2Net_backbone(
    filters,
    num_blocks,
    width_ratio,
    scale_ratio,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        f"stage1.block{num_blocks[0]}.post_activ",
        f"stage2.block{num_blocks[1]}.post_activ",
        f"stage3.block{num_blocks[2]}.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        width_ratio=width_ratio,
        scale_ratio=scale_ratio,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net50(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 6, 3],
        width_ratio=26,
        scale_ratio=4,
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


def Res2Net50_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block6.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net50,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


Res2Net50_26w4s = Res2Net50
Res2Net50_26w4s_backbone = Res2Net50_backbone


def Res2Net50_26w6s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 6, 3],
        width_ratio=26,
        scale_ratio=6,
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


def Res2Net50_26w6s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block6.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net50_26w6s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net50_26w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 6, 3],
        width_ratio=26,
        scale_ratio=8,
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


def Res2Net50_26w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block6.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net50_26w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net50_48w2s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 6, 3],
        width_ratio=48,
        scale_ratio=2,
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


def Res2Net50_48w2s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block6.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net50_48w2s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net50_14w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 6, 3],
        width_ratio=14,
        scale_ratio=8,
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


def Res2Net50_14w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block6.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net50_14w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def Res2Net101(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 23, 3],
        width_ratio=26,
        scale_ratio=4,
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


def Res2Net101_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block23.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net101,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


Res2Net101_26w4s = Res2Net101
Res2Net101_26w4s_backbone = Res2Net101_backbone


def Res2Net101_26w6s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 23, 3],
        width_ratio=26,
        scale_ratio=6,
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


def Res2Net101_26w6s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block23.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net101_26w6s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net101_26w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 23, 3],
        width_ratio=26,
        scale_ratio=8,
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


def Res2Net101_26w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block23.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net101_26w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net101_48w2s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 23, 3],
        width_ratio=48,
        scale_ratio=2,
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


def Res2Net101_48w2s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block23.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net101_48w2s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net101_14w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 4, 23, 3],
        width_ratio=14,
        scale_ratio=8,
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


def Res2Net101_14w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block4.post_activ",
        "stage3.block23.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net101_14w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net152(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 8, 36, 3],
        width_ratio=26,
        scale_ratio=4,
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


def Res2Net152_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block8.post_activ",
        "stage3.block26.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net152,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

Res2Net152_26w4s = Res2Net152
Res2Net152_26w4s_backbone = Res2Net152_backbone

def Res2Net152_26w6s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 8, 36, 3],
        width_ratio=26,
        scale_ratio=6,
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


def Res2Net152_26w6s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block8.post_activ",
        "stage3.block36.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net152_26w6s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net152_26w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 8, 36, 3],
        width_ratio=26,
        scale_ratio=8,
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


def Res2Net152_26w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block8.post_activ",
        "stage3.block36.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net152_26w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net152_48w2s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 8, 36, 3],
        width_ratio=48,
        scale_ratio=2,
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


def Res2Net152_48w2s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block8.post_activ",
        "stage3.block36.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net152_48w2s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def Res2Net152_14w8s(
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
    
    model = Res2Net(
        filters=64,
        num_blocks=[4, 8, 36, 3],
        width_ratio=14,
        scale_ratio=8,
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


def Res2Net152_14w8s_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.post_activ",
        "stage2.block8.post_activ",
        "stage3.block36.post_activ",
    ]

    return create_model_backbone(
        model_fn=Res2Net152_14w8s,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
