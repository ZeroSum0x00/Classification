"""
    gMLP: MLP-based Backbone with Gated Channel-Spatial Modulation
    
    Overview:
        Gated MLP (gMLP) is a novel neural architecture that removes self-attention
        and convolutions, using only Multi-Layer Perceptrons (MLPs) enhanced by spatial
        gating units. It achieves competitive performance in vision tasks through 
        lightweight computation and strong inductive bias via gating.
    
        gMLP serves as an alternative backbone to transformers and CNNs, showing that
        pure MLPs, when properly structured, can encode spatial interactions effectively.
    
        Key innovations include:
            - Spatial Gating Unit (SGU): Enables spatial mixing across tokens
            - Linear-Only Blocks: Fully connected layers for both token and channel mixing
            - Lightweight and Parallelizable: Efficient for large-scale training
    
    Key Components:
        • Patch Embedding:
            - Input image is divided into non-overlapping patches (e.g., 16×16), then flattened.
            - A linear projection embeds each patch to a fixed-dimensional token vector.
    
        • gMLP Block:
            - Core building block composed of:
                1. **Linear Layer** (Channel Projection): Expands channel width (token-wise MLP)
                2. **GELU Activation**
                3. **Spatial Gating Unit (SGU)**:
                    • Splits the feature along channels into two halves:  
                      - One half is left as is  
                      - The other half is passed through a depthwise linear layer (across tokens)
                    • The two halves are multiplied elementwise → enables spatial interactions
                4. **Final Linear Layer** (Channel Mixing)
    
        • Residual Connection:
            - Each gMLP block uses a residual connection:  
              `Output = Input + gMLPBlock(Input)`
    
        • Model Structure:
            - gMLP is built by stacking multiple identical gMLP blocks.
            - No convolutions or attention modules are used.
            - Final classification is done via global average pooling and MLP head.

        • Scalability:
            - Works well when scaled (e.g., gMLP-S, gMLP-B, gMLP-XL).
            - Parallelizable and efficient for large datasets.

    Model Parameter Comparison:
       -----------------------------------------
      |       Model Name      |    Params       |
      |-----------------------------------------|
      |      gMLP-tiny-16     |     5,867,328   |
      |-----------------------------------------|
      |      gMLP-small-16    |    19,422,656   |
      |-----------------------------------------|
      |      gMLP-base-16     |    73,075,392   |
       -----------------------------------------

    References:
        - Paper: “Pay Attention to MLPs”  
          https://arxiv.org/abs/2105.08050
    
        - Official implementation (Google Research):  
          https://github.com/google-research/gmlp

        - TensorFlow/Keras implementation:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/gated_mlp.py
          
        - PyTorch implementation:  
          https://github.com/rishikksh20/gMLP-pytorch

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout,
    Reshape, Permute, GlobalAveragePooling1D,
    add, multiply,
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    SplitWrapper,
)
from utils.model_processing import process_model_input, check_regularizer



def spatial_gating_block(
    inputs,
    activation=None,
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"spatial_gating_block_{K.get_uid('spatial_gating_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    xx, yy = SplitWrapper(num_or_size_splits=2, axis=-1, name=f"{name}.split")(inputs)
    
    yy = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Permute(dims=(2, 1)),
        Dense(
            units=yy.shape[1],
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        Permute(dims=(2, 1)),
    ], name=f"{name}.yy")(yy)

    gated_out = multiply([xx, yy], name=f"{name}.multiply")
    return gated_out


def res_gated_mlp_block(
    inputs,
    channels_mlp_dim,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"res_gated_mlp_block_{K.get_uid('res_gated_mlp_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        Dense(
            units=channels_mlp_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_activation_from_name(activation),
    ], name=f"{name}.project")(inputs)

    x = spatial_gating_block(
        inputs=x,
        activation=None,
        normalizer=normalizer,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        name=f"{name}.gating_block"
    )
    
    x = Dense(
        units=inputs.shape[-1],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.dense2"
    )(x)

    if drop_rate > 0:
        x = Dropout(drop_rate, noise_shape=(None, 1, 1), name=f"{name}.dropout")(x)
        
    return add([x, inputs], name=f"{name}.add")

    
def gMLP(
    stem_width,
    patch_size,
    num_blocks,
    channels_mlp_dim,
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
        
        x = res_gated_mlp_block(
            inputs=x,
            channels_mlp_dim=channels_mlp_dim,
            **layer_constant_dict,
            drop_rate=block_drop_rate,
            name=f"stage{i + 1}"
        )
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage{i + 1}.final_norm")(x)
             
    if include_head:
        x = Sequential([
            GlobalAveragePooling1D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "gMLP"
    if stem_width == 128:
        model_name += "-tiny"
    elif stem_width == 256:
        model_name += "-small"
    elif stem_width == 512:
        model_name += "-base"
    model_name += f"-{patch_size}"
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def gMLP_T16(
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

    model = gMLP(
        stem_width=128,
        patch_size=16,
        num_blocks=30,
        channels_mlp_dim=128*6,
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


def gMLP_S16(
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
        
    model = gMLP(
        stem_width=256,
        patch_size=16,
        num_blocks=30,
        channels_mlp_dim=256*6,
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


def gMLP_B16(
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
    
    model = gMLP(
        stem_width=512,
        patch_size=16,
        num_blocks=30,
        channels_mlp_dim=512*6,
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
    