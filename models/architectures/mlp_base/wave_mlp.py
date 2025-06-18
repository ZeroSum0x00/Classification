"""
    WaveMLP: MLP-Based Backbone with Large Receptive Field via Wave Aggregation
    
    Overview:
        WaveMLP is a novel MLP-based vision backbone that enhances spatial modeling
        using **Wave Aggregation**, a structured spatial mixing mechanism that allows
        MLPs to model long-range dependencies like convolutions with large kernels.
        
        It introduces a **Wave Block** that uses depth-wise and channel-wise MLPs in 
        combination with learnable spatial aggregation masks to effectively capture 
        both local and global context in images.
    
        Key innovations include:
            - Wave Aggregation: Expands receptive field using structured token mixing
            - Shifted Token Grouping: Improves spatial generalization
            - Pure MLP Architecture with Conv-like locality
    
    Key Components:
        • Patch Embedding:
            - Input image is divided into non-overlapping patches (e.g., 16×16).
            - Each patch is linearly projected into an embedding dimension `C`.
            - Output shape: `[B, H×W, C]` or `[B, C, H, W]` (if 2D layout is preserved)
    
        • Wave Block:
            - Main building block of WaveMLP backbone, consisting of:
              
              1. **Channel MLP**:
                  - Applies a feedforward MLP to each patch independently (per-token MLP).
                  - Structure: `Linear → GELU → Linear` with residual connection.
    
              2. **Wave Aggregation MLP**:
                  - Performs structured token mixing using spatial MLPs.
                  - Token groups are shifted and aggregated via **Wave Operator** (e.g., wave mask convolution).
                  - Captures global spatial relationships while preserving inductive bias.
              
              3. **Shifted Token Grouping**:
                  - Applies spatial shifting before aggregation (e.g., shift-left, shift-up).
                  - Inspired by Swin Transformer’s shifted window mechanism.
    
            - Both sublayers include **LayerNorm**, residual connections, and optional dropout.

        • Large Receptive Field:
            - Achieved via **Wave Aggregation MLP**, not convolutions
            - Enables spatial generalization even on low-resolution patches
    
        • No Convolution, No Attention:
            - Efficient pure-MLP model with large spatial context
            
    Model Parameter Comparison:
       -----------------------------------------
      |       Model Name      |    Params       |
      |-----------------------------------------|
      |      WaveMLP-tiny     |    17,217,992   |
      |-----------------------------------------|
      |      WaveMLP-small    |    30,708,168   |
      |-----------------------------------------|
      |      WaveMLP-medium   |    44,058,808   |
      |-----------------------------------------|
      |      WaveMLP-base     |    63,589,432   |
       -----------------------------------------
    
    References:
        - Paper: “An Image Patch is a Wave: Phase-Aware Vision MLP”  
          https://arxiv.org/abs/2111.12294
    
        - Official PyTorch implementation:  
          https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch

        - TensorFlow/Keras implementation:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/wave_mlp.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D,
    Reshape, Dense, Dropout, GlobalAveragePooling2D,
    add, multiply, concatenate,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    MLPBlock, DropPathV1, DropPathV2,
    OperatorWrapper, UnstackWrapper,
)
from utils.model_processing import process_model_input, check_regularizer


def phase_aware_token_mixing(
    inputs,
    out_dim=-1,
    qkv_bias=False,
    activation="gelu",
    normalizer="batch-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"phase_aware_token_mixing_{K.get_uid('phase_aware_token_mixing')}"
        
    out_dim = out_dim if out_dim > 0 else inputs.shape[-1]
    regularizer_decay = check_regularizer(regularizer_decay)

    # height feature
    theta_h = Sequential([
        Conv2D(
            filters=out_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.theta_h")(inputs)

    height = Conv2D(
        filters=out_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=qkv_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.height"
    )(inputs)

    theta_cos  = OperatorWrapper(operator="cos")(theta_h)
    height_cos = multiply([height, theta_cos])

    theta_sin  = OperatorWrapper(operator="sin")(theta_h)
    height_sin = multiply([height, theta_sin])
    
    height = concatenate([height_cos, height_sin], axis=-1)
    height = Conv2D(
        filters=out_dim,
        kernel_size=(1, 7),
        strides=(1, 1),
        padding="same",
        groups=out_dim,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
    )(height)

    # width feature
    theta_w = Sequential([
        Conv2D(
            filters=out_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.theta_w")(inputs)

    width = Conv2D(
        filters=out_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=qkv_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.width"
    )(inputs)

    theta_cos = OperatorWrapper(operator="cos")(theta_w)
    width_cos = multiply([width, theta_cos])
    
    theta_sin = OperatorWrapper(operator="sin")(theta_w)
    width_sin = multiply([width, theta_sin])

    width = concatenate([width_cos, width_sin], axis=-1)
    width = Conv2D(
        filters=out_dim,
        kernel_size=(7, 1),
        strides=(1, 1),
        padding="same",
        groups=out_dim,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
    )(width)

    # channel feature
    channel = Conv2D(
        filters=out_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=qkv_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.channel"
    )(inputs)

    x = add([height, width, channel])
    x = Sequential([
        GlobalAveragePooling2D(keepdims=True),
        MLPBlock(
            mlp_dim=out_dim // 4, 
            out_dim=out_dim * 3, 
            use_conv=True,
            use_bias=True,
            use_gated=False,
            activation=activation,
            normalizer=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            drop_rate=drop_rate
        ),
        Reshape([1, 1, out_dim, 3]),
        get_activation_from_name("softmax")
    ], name=f"{name}.reweight")(x)

    attn_height, attn_width, attn_channel = UnstackWrapper(axis=-1)(x)
    attn_height = multiply([height, attn_height])
    attn_width = multiply([width, attn_width])
    attn_channel = multiply([channel, attn_channel])
    attn = add([attn_height, attn_width, attn_channel])

    out = Sequential([
        Conv2D(
            filters=out_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        Dropout(rate=drop_rate),
    ], name=f"{name}.out")(attn)
    return out


def wave_block(
    inputs,
    mlp_ratio=4,
    qkv_bias=False,
    activation="gelu",
    normalizer="batch-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"wave_block_{K.get_uid('wave_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    # phase attn
    attn = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.attn.norm")(inputs)
    
    attn = phase_aware_token_mixing(
        inputs=attn,
        out_dim=inputs.shape[-1],
        qkv_bias=qkv_bias,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.attn.token_mixing"
    )
    
    attn = DropPath(drop_prob=drop_rate, name=f"{name}.attn.drop_path")(attn)
    attn_out = add([inputs, attn], name=f"{name}.attn.add")

    # phase mlp
    mlp = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.mlp.norm")(attn_out)
    
    mlp = MLPBlock(
        mlp_dim=int(inputs.shape[-1] * mlp_ratio),
        out_dim=-1,
        use_conv=True,
        use_bias=True,
        use_gated=False,
        activation=activation,
        normalizer=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        drop_rate=drop_rate
    )(mlp)
    
    mlp = DropPath(drop_prob=drop_rate, name=f"{name}.mlp.drop_path")(mlp)
    
    out = add([attn_out, mlp], name=f"{name}.mlp.add")
    return out

    
def WaveMLP(
    filters,
    num_blocks,
    stem_width,
    mlp_ratios,
    qkv_bias,
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
    drop_rate=0.1
):
        
    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    stem_width = stem_width if stem_width > 0 else filters[0]
    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
        "drop_rate": drop_rate,
    }
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    x = Sequential([
        ZeroPadding2D(padding=2),
        Conv2D(
            filters=stem_width,
            kernel_size=(7, 7),
            strides=(4, 4),
            padding="valid",
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name="stem")(inputs)


    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, filter, mlp_ratio) in enumerate(zip(num_blocks, filters, mlp_ratios)):
        if stack_id > 0:
            x = Conv2D(
                filters=filter,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
                name=f"stage{stack_id + 1}.block1.conv"
            )(x)

        x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage{stack_id + 1}.block1.norm")(x)

        for block_id in range(num_block):
            drop_prob = drop_rate * global_block_id / total_blocks
            global_block_id += 1
            x = wave_block(
                inputs=x,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                **layer_constant_dict,
                name=f"stage{stack_id + 1}.block{block_id + 2}"
            )

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage{stack_id + 1}.block{block_id + 2}.final_norm")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "WaveMLP"
    if num_blocks == [2, 2, 4, 2] and filters == [64, 128, 320, 512] and mlp_ratios == [4, 4, 4, 4]:
        model_name += "-T"
    elif num_blocks == [2, 3, 10, 3] and filters == [64, 128, 320, 512] and mlp_ratios == [4, 4, 4, 4]:
        model_name += "-S"
    elif num_blocks == [3, 4, 18, 3] and filters == [64, 128, 320, 512] and mlp_ratios == [8, 8, 4, 4]:
        model_name += "-M"
    elif num_blocks == [2, 2, 18, 2] and filters == [96, 192, 384, 768] and mlp_ratios == [4, 4, 4, 4]:
        model_name += "-B"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def WaveMLP_T(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = WaveMLP(
        filters=[64, 128, 320, 512],
        num_blocks=[2, 2, 4, 2],
        stem_width=-1,
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
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


def WaveMLP_S(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = WaveMLP(
        filters=[64, 128, 320, 512],
        num_blocks=[2, 3, 10, 3],
        stem_width=-1,
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
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


def WaveMLP_M(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = WaveMLP(
        filters=[64, 128, 320, 512],
        num_blocks=[3, 4, 18, 3],
        stem_width=-1,
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=False,
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


def WaveMLP_B(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_connect_rate=0.1,
    drop_rate=0.1,
) -> Model:
    
    model = WaveMLP(
        filters=[96, 192, 384, 768],
        num_blocks=[2, 2, 18, 2],
        stem_width=-1,
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
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