"""
    ResMLP: Residual MLP-Only Backbone with Linear Token Mixing
    
    Overview:
        ResMLP is a pure MLP-based vision backbone that introduces simple residual
        connections and linear token mixing to build a deep network without convolutions
        or attention. It separates spatial and feature mixing through linear layers,
        relying on large-scale data and architectural simplicity to achieve strong results.
    
        Key innovations include:
            - Linear Token Mixing: Projects and mixes token (patch) information linearly
            - Channel-wise MLP: Processes each patch independently in feature space
            - Residual Pre-activation and LayerNorm: Stabilize training and deep stacking
    
    Key Components:
        • Patch Embedding:
            - The image is divided into fixed-size patches (e.g., 16×16).
            - Each patch is flattened and projected using a linear layer to a fixed dimension `d`.
            - Output shape: `[B, N, d]` where `N = num_patches`, `d = embedding_dim`.
    
        • ResMLP Block:
            - Consists of two main sub-layers applied with residual connections:
            
              1. **Token Mixing Linear Layer**:
                  - Transpose input to shape `[B, d, N]`
                  - Apply a fully-connected linear layer along token dimension (N)
                  - Mixes spatial relationships between patches
                  - Transpose back to `[B, N, d]`
    
              2. **Channel MLP**:
                  - Two-layer MLP with GELU activation
                  - Applies independently to each patch vector `[d]`
                  - Form: `MLP(x) = Linear → GELU → Linear`
    
            - Each sub-layer is followed by:
                - LayerNorm (pre-activation)
                - Residual connection

        • Training Stability:
            - Uses **pre-norm** (LayerNorm before each layer) and **deep residuals**
            - Easy to scale to dozens of layers
    
        • No attention, no convolutions:
            - Enables fully linear models suitable for hardware optimization

    Model Parameter Comparison:
       ---------------------------------------------
      |        Model Name         |    Params       |
      |---------------------------------------------|
      |      ResMLP-small-12      |    15,350,872   |
      |---------------------------------------------|
      |      ResMLP-small-24      |    30,020,680   |
      |---------------------------------------------|
      |      ResMLP-small-36      |    44,690,488   |
      |---------------------------------------------|
      |      ResMLP-base-24       |   115,736,776   |
       ---------------------------------------------

    References:
        - Paper: “ResMLP: Feedforward networks for image classification with data-efficient training”  
          https://arxiv.org/abs/2105.03404
    
        - Official code (Facebook Research):  
          https://github.com/facebookresearch/deit

        - TensorFlow/Keras implementation:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/res_mlp.py
          
        - PyTorch implementation:  
          https://github.com/rishikksh20/ResMLP-pytorch
          
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Reshape, Permute,
    Dense, Dropout, GlobalAveragePooling1D,
    add,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    ChannelAffine,
)
from utils.model_processing import process_model_input, check_regularizer


def res_mlp_block(
    inputs,
    channels_dim,
    activation="gelu",
    normalizer=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    name=None
):
    if name is None:
        name = f"res_mlp_block_{K.get_uid('res_mlp_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
    
    x = Sequential([
        ChannelAffine(use_bias=True, weight_init_value=-1, axis=-1),
        Permute(dims=(2, 1)),
        Dense(
            units=nn.shape[-1],
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        Permute(dims=(2, 1)),
        ChannelAffine(use_bias=False, weight_init_value=-1, axis=-1),
        Dropout(
            rate=drop_rate,
            noise_shape=(None, 1, 1),
        ),
    ], name=f"{name}.token_mixing")(inputs)

    token_out = add([inputs, x], name=f"{name}.merge_token_mixing")

    x = Sequential([
        ChannelAffine(use_bias=True, weight_init_value=-1, axis=-1),
        Dense(
            units=channels_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Dense(
            units=inputs.shape[-1],
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        ChannelAffine(use_bias=False, weight_init_value=-1, axis=-1),
        Dropout(
            rate=drop_rate,
            noise_shape=(None, 1, 1),
        ),
    ], name=f"{name}.channel_mixing")(token_out)
    
    out = add([x, token_out], name=f"{name}.merge_channel_mixing")
    return out

    
def ResMLP(
    stem_width,
    patch_size,
    num_blocks,
    channels_dim,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.,
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

    x = Sequential([
        Conv2D(
            filters=stem_width,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        Reshape(target_shape=(-1, stem_width)),
    ], name="stem")(inputs)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_rate, (list, tuple)) else [drop_rate, drop_rate]
    
    for i in range(num_blocks):
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * i / num_blocks
        x = res_mlp_block(
            inputs=x,
            channels_dim=channels_dim,
            **layer_constant_dict,
            drop_rate=block_drop_rate,
            name=f"stage{i + 1}"
        )
        
    x = ChannelAffine(weight_init_value=-1, axis=-1, name=f"stage{i + 1}.channel_affine")(x)
    
    if include_top:
        x = Sequential([
            GlobalAveragePooling1D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "ResMLP"
    if stem_width == 384:
        model_name += "-S"
    elif stem_width == 768:
        model_name += "-B"
    model_name += f"-{num_blocks}"
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ResMLP_S12(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = ResMLP(
        stem_width=384,
        patch_size=16,
        num_blocks=12,
        channels_mlp_dim=384 * 4,
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
        drop_connect_rate=drop_connect_rate,
        drop_rate=drop_rate
    )
    return model


def ResMLP_S24(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
        
    model = ResMLP(
        stem_width=384,
        patch_size=16,
        num_blocks=24,
        channels_mlp_dim=384 * 4,
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
        drop_connect_rate=drop_connect_rate,
        drop_rate=drop_rate
    )
    return model


def ResMLP_S36(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = ResMLP(
        stem_width=384,
        patch_size=16,
        num_blocks=36,
        channels_mlp_dim=384 * 4,
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
        drop_connect_rate=drop_connect_rate,
        drop_rate=drop_rate
    )
    return model


def ResMLP_B24(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = ResMLP(
        stem_width=768,
        patch_size=8,
        num_blocks=24,
        channels_mlp_dim=768 * 4,
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
        drop_connect_rate=drop_connect_rate,
        drop_rate=drop_rate
    )
    return model
    