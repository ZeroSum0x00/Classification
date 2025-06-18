"""
    ConvNeXt V2: Scaling Up and Adding Inverse Depths with Global Response Normalization
    
    Overview:
        ConvNeXt V2 is the improved version of ConvNeXt that combines the strengths of
        convolutional networks and transformer-era insights with **further enhancements**:
          - **Global Response Normalization (GRN)** for better generalization
          - **Stronger scaling** with inverted stage depth design (e.g., 9-9-3-3)
          - **Large-scale training** (up to 4K image size, 64K batch)
          - State-of-the-art performance on classification, detection, segmentation, and foundation model pretraining.
    
        ConvNeXt V2 demonstrates that *pure CNNs* can match or outperform Transformers
        at scale, while maintaining deployment simplicity and high throughput.
    
    Key Components:
        • Patchify Stem:
            - Uses a 4×4 conv with stride 4 to embed non-overlapping image patches
            - Converts image to token-like 1D sequence (similar to ViT)
    
        • ConvNeXt V2 Block:
            - Builds on ConvNeXt block but adds **Global Response Normalization (GRN)**
            - **GRN**:
                - A normalization across spatial locations to stabilize training:
                  ```
                  GRN(x) = γ * (x / ||x||_2) + β
                  ```
                - Helps improve robustness during large-scale and foundation pretraining
    
        • Downsampling Layers:
            - 2×2 Conv with stride 2 between stages to reduce spatial resolution
    
        • Inverted Stage Depth Design:
            - Reverses the standard (3-3-9-3) layout into a top-heavy structure like (9-9-3-3)
            - More early-stage capacity leads to better large-scale performance
    
        • Architecture Variants:
            - ConvNeXt V2-Tiny, Small, Base, Large, Huge
            - Trained with or without masked autoencoding (MAE-style)
            - Foundation models trained with 2K–4K resolution and 64K batch
    
        • Efficiency and Scaling:
            - GRN improves training and convergence at large scale
            - ConvNeXt V2 models can outperform ViT/MAE with fewer FLOPs

    General Model Architecture (ConvNeXt-v2 base example):
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
         ---------------------------------------------
        |        Model Name         |      Params     |
        |---------------------------------------------|
        |     ConvNeXt-v2-atto      |     3,708,400   |
        |---------------------------------------------|
        |     ConvNeXt-v2-femto     |     5,233,240   |
        |---------------------------------------------|
        |     ConvNeXt-v2-pico      |     9,066,280   |
        |---------------------------------------------|
        |     ConvNeXt-v2-nano      |    15,623,800   |
        |---------------------------------------------|
        |     ConvNeXt-v2-tiny      |    28,635,496   |
        |---------------------------------------------|
        |     ConvNeXt-v2-base      |    88,717,800   |
        |---------------------------------------------|
        |     ConvNeXt-v2-large     |   197,956,840   |
        |---------------------------------------------|
        |     ConvNeXt-v2-huge      |   660,289,640   |
         ---------------------------------------------

    References:
        - Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
          https://arxiv.org/pdf/2301.00808.pdf
          
        - Official PyTorch repository:
          https://github.com/facebookresearch/ConvNeXt-V2
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/convnext
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential

from .convnext import convnext_original_block, convnext_kecam_block, ConvNeXt
from utils.model_processing import (
    process_model_input, create_model_backbone, check_regularizer,
)


def ConvNeXt_v2(
    block,
    filters,
    num_blocks,
    layer_scale_init_value=0.,
    use_grn=True,
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
    
    backbone = ConvNeXt(
        block=block,
        filters=filters,
        num_blocks=num_blocks,
        layer_scale_init_value=layer_scale_init_value,
        use_grn=use_grn,
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
    
    model_name = "ConvNeXt-v2"
    if num_blocks == [2, 2, 6, 2]:
        if filters == 40:
            model_name += "-atto"
        elif filters == 48:
            model_name += "-femto"
        elif filters == 64:
            model_name += "-pico"
    elif filters == 80 and num_blocks == [2, 2, 8, 2]:
        model_name += "-nano"
    elif filters == 96 and num_blocks == [3, 3, 9, 3]:
        model_name += "-tiny"
    elif num_blocks == [3, 3, 27, 3]:
        if filters ==128:
            model_name += "-base"
        elif filters ==192:
            model_name += "-large"
        elif filters == 352:
            model_name += "-huge"

    if block == convnext_original_block:
        model_name += "[Original]"
    elif block == convnext_kecam_block:
        model_name += "[Kecam]"
        
    model = Model(inputs=inputs, outputs=backbone.outputs, name=model_name)
    return model


def ConvNeXt_v2_backbone(
    block,
    filters,
    num_blocks,
    layer_scale_init_value=0.,
    use_grn=True,
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
        model_fn=ConvNeXt_v2,
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


def ConvNeXt_v2_atto(
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
    
    model = ConvNeXt_v2(
        block=block,
        filters=40,
        num_blocks=[2, 2, 6, 2],
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


def ConvNeXt_v2_atto_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_atto,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_atto(
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
    
    model = ConvNeXt_v2_atto(
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


def ConvNeXt_v2_original_atto_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_original_atto,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_atto(
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
    
    model = ConvNeXt_v2_atto(
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


def ConvNeXt_v2_kecam_atto_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_kecam_atto,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_femto(
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
    
    model = ConvNeXt_v2(
        block=block,
        filters=48,
        num_blocks=[2, 2, 6, 2],
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


def ConvNeXt_v2_femto_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_femto,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_femto(
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
    
    model = ConvNeXt_v2_femto(
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


def ConvNeXt_v2_original_femto_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_original_femto,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_femto(
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
    
    model = ConvNeXt_v2_femto(
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


def ConvNeXt_v2_kecam_femto_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_kecam_femto,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_pico(
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
    
    model = ConvNeXt_v2(
        block=block,
        filters=64,
        num_blocks=[2, 2, 6, 2],
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


def ConvNeXt_v2_pico_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_pico,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_pico(
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
    
    model = ConvNeXt_v2_pico(
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


def ConvNeXt_v2_original_pico_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_original_pico,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_pico(
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
    
    model = ConvNeXt_v2_pico(
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


def ConvNeXt_v2_kecam_pico_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block7.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_kecam_pico,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_nano(
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
    
    model = ConvNeXt_v2(
        block=block,
        filters=80,
        num_blocks=[2, 2, 8, 2],
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


def ConvNeXt_v2_nano_backbone(
    block,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block9.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_nano,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_nano(
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
    
    model = ConvNeXt_v2_nano(
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


def ConvNeXt_v2_original_nano_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block9.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_original_nano,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_nano(
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
    
    model = ConvNeXt_v2_nano(
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


def ConvNeXt_v2_kecam_nano_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stage1.block2.final",
        "stage2.block3.final",
        "stage3.block9.final",
    ]

    return create_model_backbone(
        model_fn=ConvNeXt_v2_kecam_nano,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_tiny(
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
    
    model = ConvNeXt_v2(
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


def ConvNeXt_v2_tiny_backbone(
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
        model_fn=ConvNeXt_v2_tiny,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_tiny(
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
    
    model = ConvNeXt_v2_tiny(
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


def ConvNeXt_v2_original_tiny_backbone(
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
        model_fn=ConvNeXt_v2_original_tiny,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_tiny(
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
    
    model = ConvNeXt_v2_tiny(
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


def ConvNeXt_v2_kecam_tiny_backbone(
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
        model_fn=ConvNeXt_v2_kecam_tiny,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_base(
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
    
    model = ConvNeXt_v2(
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


def ConvNeXt_v2_base_backbone(
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
        model_fn=ConvNeXt_v2_base,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_base(
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
    
    model = ConvNeXt_v2_base(
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


def ConvNeXt_v2_original_base_backbone(
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
        model_fn=ConvNeXt_v2_original_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_base(
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
    
    model = ConvNeXt_v2_base(
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


def ConvNeXt_v2_kecam_base_backbone(
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
        model_fn=ConvNeXt_v2_kecam_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_large(
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
    
    model = ConvNeXt_v2(
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


def ConvNeXt_v2_large_backbone(
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
        model_fn=ConvNeXt_v2_large,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_large(
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
    
    model = ConvNeXt_v2_large(
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


def ConvNeXt_v2_original_large_backbone(
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
        model_fn=ConvNeXt_v2_original_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_large(
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
    
    model = ConvNeXt_v2_large(
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


def ConvNeXt_v2_kecam_large_backbone(
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
        model_fn=ConvNeXt_v2_kecam_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_huge(
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
    
    model = ConvNeXt_v2(
        block=block,
        filters=352,
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


def ConvNeXt_v2_huge_backbone(
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
        model_fn=ConvNeXt_v2_huge,
        custom_layers=custom_layers,
        block=block,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def ConvNeXt_v2_original_huge(
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
    
    model = ConvNeXt_v2_huge(
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


def ConvNeXt_v2_original_huge_backbone(
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
        model_fn=ConvNeXt_v2_original_huge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    

def ConvNeXt_v2_kecam_huge(
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
    
    model = ConvNeXt_v2_huge(
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


def ConvNeXt_v2_kecam_huge_backbone(
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
        model_fn=ConvNeXt_v2_kecam_huge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )