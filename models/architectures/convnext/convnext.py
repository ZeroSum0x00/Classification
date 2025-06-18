"""
    ConvNeXt: A Modernized ConvNet for Scalable Visual Recognition
    
    Overview:
        ConvNeXt is a pure convolutional backbone that revisits and modernizes
        traditional ResNet-style architectures using design insights from Transformers
        (like the Vision Transformer). It achieves state-of-the-art accuracy on
        ImageNet and competitive results on dense prediction tasks, **without self-attention**.
    
        Key innovations include:
            - Replacing BatchNorm with LayerNorm
            - Depthwise Separable Convolutions
            - Inverted bottlenecks (like in MobileNetV2)
            - Large kernel sizes (7×7 depthwise conv)
            - GELU activation and fewer activation layers
            - Transformer-style scaling (Stage depth: 3-3-9-3)
    
        ConvNeXt is scalable and efficient, with variants from tiny to huge.
    
    Key Components:
        • Patchify Stem:
            - Replaces the standard 7×7 conv with a **patchify layer**
            - Uses a 4×4 Conv2d with stride 4 to reduce spatial size at the beginning
            - Mimics ViT-style patch embedding
    
        • ConvNeXt Block:
            - Core building unit inspired by the Transformer feed-forward block
            - No shortcut MLP or attention — purely convolutional
            - All blocks use **channel-last** format for better hardware performance
    
        • Downsampling:
            - Between stages: 2×2 Conv with stride 2 to reduce spatial resolution
            - Typically 4 stages (stem + stage1 → stage4)
    
        • Architecture Design:
            - ConvNeXt follows a 4-stage macro-structure like ViT/ResNet:
              - Stage 1: Output 96 channels
              - Stage 2: Output 192 channels
              - Stage 3: Output 384 channels
              - Stage 4: Output 768 channels
            - Stage depths vary across variants (e.g., 3-3-9-3 in ConvNeXt-T)
    
        • Model Variants:
            - **ConvNeXt-T** (Tiny)
            - **ConvNeXt-S** (Small)
            - **ConvNeXt-B** (Base)
            - **ConvNeXt-L** (Large)
            - **ConvNeXt-XL / H** (for very large-scale tasks)
    
        • Advantages:
            - Pure convolutional, yet performs as well or better than ViT on many benchmarks
            - Scales well to large compute (training with 8K batch size)
            - Easy to deploy on existing CNN infrastructure

    General Model Architecture:
         ------------------------------------------------------------------------------------
        | Stage                  | Layer                            | Output Shape           |
        |------------------------+----------------------------------+------------------------|
        | Input                  | input_layer                      | (None, 224, 224, 3)    |
        |------------------------+----------------------------------+------------------------|
        | Stem                   | ConvolutionBlock (4x4, s=4, p=2) | (None, 56, 56, C)      |
        |------------------------+----------------------------------+------------------------|
        | Stage 1                | ConvNeXt_block (3x)              | (None, 55, 55, C)      |
        |------------------------+----------------------------------+------------------------|
        | Stage 2                | ConvolutionBlock (2x2, s=2, p=1) | (None, 28, 28, 2C)     |
        |                        | ConvNeXt_block (3x)              | (None, 28, 28, 2C)     |
        |------------------------+----------------------------------+------------------------|
        | Stage 3                | ConvolutionBlock (2x2, s=2, p=1) | (None, 14, 14, 4C)     |
        |                        | ConvNeXt_block (27x)             | (None, 14, 14, 4C)     |
        |------------------------+----------------------------------+------------------------|
        | Stage 4                | ConvolutionBlock (2x2, s=2, p=1) | (None, 7, 7, 8C)       |
        |                        | ConvNeXt_block (3x)              | (None, 7, 7, 8C)       |
        |------------------------+----------------------------------+------------------------|
        | CLS Logics             | GlobalAveragePooling             | (None, 8C)             |
        |                        | fc (Logics)                      | (None, 1000)           |
         ------------------------------------------------------------------------------------
         
    Model Parameter Comparison:
         ------------------------------------------
        |       Model Name       |      Params     |
        |------------------------------------------|
        |     ConvNeXt-tiny      |    28,582,504   |
        |------------------------------------------|
        |     ConvNeXt-small     |    50,210,152   |
        |------------------------------------------|
        |     ConvNeXt-base      |    88,573,416   |
        |------------------------------------------|
        |     ConvNeXt-large     |   197,740,264   |
        |------------------------------------------|
        |     ConvNeXt-xlarge    |   350,160,872   |
        |------------------------------------------|
        |     ConvNeXt-xxlarge   |   846,411,496   |
         ------------------------------------------

    Notes:
        - Parameter counts are taken from both the official PyTorch implementation and
          the TensorFlow-converted version (Keras-based).
        - Minor discrepancies in parameters may occur depending on layer conversion strategies,
          especially with normalization and weight initialization.
        - Input shape: (224, 224, 3) is the standard test size used in the original paper.

    References:
        - Paper: "A ConvNet for the 2020s"
          https://arxiv.org/pdf/2201.03545.pdf
          
        - Official PyTorch repository:
          https://github.com/facebookresearch/ConvNeXt
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/convnext/convnext.py
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, Dense, Dropout,
    Lambda, GlobalAveragePooling2D,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    StochasticDepthV1, StochasticDepthV2,
    LayerScaleAndDropBlock, LinearLayer,
)
from utils.model_processing import (
    process_model_input, create_model_backbone,
    create_layer_instance, check_regularizer,
)



def convnext_original_block(
    inputs,
    layer_scale_init_value=1e-6,
    use_grn=False,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_prob=0.1,
    name=None
):
    if name is None:
        name = f"convnext_original_block_{K.get_uid('convnext_original_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
  
    in_filters = inputs.shape[-1]

    x = Sequential([
        Conv2D(
            filters=in_filters,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="same",
            groups=in_filters,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
            Conv2D(
            filters=in_filters * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
        get_normalizer_from_name("global-response-norm", axis=-1, epsilon=norm_eps) if use_grn else LinearLayer(),
        Conv2D(
            filters=in_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
    ], name=f"{name}.conv_block")(inputs)

    if layer_scale_init_value > 0:
        layer_scale_gamma = tf.ones(in_filters) * layer_scale_init_value
        x = x * layer_scale_gamma

    if drop_prob > 0:
        x = StochasticDepthV1(drop_prob, name=f"{name}.drop_path")([inputs, x])
    
    x = Lambda(lambda x: x, name=f"{name}.final")(x)
    return x


def convnext_kecam_block(
    inputs,
    layer_scale_init_value=1e-6,
    use_grn=False,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_prob=0.1,
    name=None
):
    if name is None:
        name = f"convnext_kecam_block_{K.get_uid('convnext_kecam_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
  
    in_filters = inputs.shape[-1]

    x = Sequential([
        DepthwiseConv2D(
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="same",
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        Dense(
            units=in_filters * 4,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
        get_normalizer_from_name("global-response-norm", axis=-1, epsilon=norm_eps) if use_grn else LinearLayer(),
        Dense(
            units=in_filters,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        )
    ], name=f"{name}.conv_block")(inputs)

    x = LayerScaleAndDropBlock(layer_scale=layer_scale_init_value, drop_rate=drop_prob, name=f"{name}.scale_and_drop_block")([inputs, x])
    x = Lambda(lambda x: x, name=f"{name}.final")(x)
    return x


def ConvNeXt(
    block,
    filters,
    num_blocks,
    layer_scale_init_value=1e-6,
    use_grn=False,
    channel_scale=2,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
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
        weights=weights
    )
    
    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks))]

    current_id = 0
    dp_rates = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(num_blocks))]

    x = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name="stem")(inputs)

    for i in range(len(num_blocks)):
        if i > 0:
            x = Sequential([
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                Conv2D(
                    filters=filters[i],
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=regularizer_decay,
                ),
            ], name=f"stage{i + 1}.block1")(x)

        for j in range(num_blocks[i]):
            stage_name = f"stage{i + 1}.block{j + 2}" if i > 0 else f"stage{i + 1}.block{j + 1}"
            x = create_layer_instance(
                block,
                inputs=x,
                drop_prob=dp_rates[current_id + j],
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                **layer_constant_dict,
                name=stage_name
            )
            
        current_id += num_blocks[i]

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            Dense(
                units=1 if num_classes == 2 else num_classes,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "ConvNeXt"
    if filters[0] == 96 and num_blocks == [3, 3, 9, 3]:
        model_name += "-tiny"
    elif num_blocks == [3, 3, 27, 3]:
        if filters[0] ==96:
            model_name += "-small"
        elif filters[0] ==128:
            model_name += "-base"
        elif filters[0] ==192:
            model_name += "-large"
        elif filters[0] == 256:
            model_name += "-xlarge"
    elif filters[0] == 384 and num_blocks == [3, 4, 30, 3]:
            model_name += "-xxlarge"

    if block == convnext_original_block:
        model_name += "[Original]"
    elif block == convnext_kecam_block:
        model_name += "[Kecam]"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ConvNeXt_backbone(
    block,
    filters,
    num_blocks,
    layer_scale_init_value=1e-6,
    use_grn=False,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        f"stage1.block{num_blocks[0]}.final",
        f"stage2.block{num_blocks[1] + 1}.final",
        f"stage3.block{num_blocks[2] + 1}.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt,
        custom_layers=custom_layers,
        block=block,
        filters=filters,
        num_blocks=num_blocks,
        layer_scale_init_value=layer_scale_init_value,
        use_grn=use_grn,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_tiny(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=96,
        num_blocks=[3, 3, 9, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_tiny_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block10.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_tiny,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_tiny(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_tiny(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_tiny_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block10.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_tiny,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_kecam_tiny(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_tiny(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_tiny_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block10.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_tiny,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_small(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=96,
        num_blocks=[3, 3, 27, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_small_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_small,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_small(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_small(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_small_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_small,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_kecam_small(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_small(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_small_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_small,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_base(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=128,
        num_blocks=[3, 3, 27, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_base_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_base,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_base(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_base(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_kecam_base(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_base(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_large(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=192,
        num_blocks=[3, 3, 27, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_large_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_large,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_large(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_large(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_large_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def ConvNeXt_kecam_large(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_large(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_large_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_xlarge(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=256,
        num_blocks=[3, 3, 27, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_xlarge_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_xlarge,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_xlarge(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_xlarge(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_xlarge_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_xlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def ConvNeXt_kecam_xlarge(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_xlarge(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_xlarge_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_xlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_xxlarge(
    block,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt(
        block=block,
        filters=384,
        num_blocks=[3, 4, 30, 3],
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_xxlarge_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block5.final",
        "stage3.block31.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_xxlarge,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_original_xxlarge(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_xxlarge(
        block=convnext_original_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_original_xxlarge_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block5.final",
        "stage3.block31.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_original_xxlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def ConvNeXt_kecam_xxlarge(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
) -> Model:
    
    model = ConvNeXt_xxlarge(
        block=convnext_kecam_block,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNeXt_kecam_xxlarge_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block5.final",
        "stage3.block31.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_kecam_xxlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    