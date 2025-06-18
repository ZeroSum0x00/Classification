"""
    Overview:
        This script provides parameter statistics for EVA (Extremely large-scale Visual pretraining with Adapter),
        a powerful vision transformer model that builds upon the BEiT framework.

        EVA incorporates advanced training techniquesand is known for its effectiveness in tasks such as image-text
        alignment, semantic segmentation, and zero-shot transfer.

    Model Parameter Comparison:
         -----------------------------------------
        |      Model Name     |      Params       |
        |---------------------+-------------------|
        |     EVA-Tiny-14     |      5,645,032    |
        |---------------------+-------------------|
        |     EVA-Small-14    |     21,905,896    |
        |---------------------+-------------------|
        |     EVA-Base-14     |     86,278,120    |
        |---------------------+-------------------|
        |     EVA-Large-14    |    303,940,584    |
        |---------------------+-------------------|
        |     EVA-Huge-14     |    631,716,840    |
        |---------------------+-------------------|
        |     EVA-Gaint-14    |  1,012,193,256    |
         -----------------------------------------

    Notes:
        - Based on the BEiT-3 and DINOv2-style architecture with advanced data augmentation and masking.
        - Parameters are from TensorFlow-converted models.
        - The "-14" suffix refers to the 14x14 patch size in the Vision Transformer backbone.
    
    References:
        - Paper: "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale"
          https://arxiv.org/pdf/2211.07636.pdf
          
        - Official PyTorch repository:
          https://github.com/baaivision/EVA/tree/master/EVA-01
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/eva.py
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from .beit import BEiT
from models.layers import get_activation_from_name
from utils.model_processing import process_model_input, check_regularizer



def EVA(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    mlp_ratio,
    attn_qkv_bias=False,
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

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    backbone = BEiT(
        vocab_size=0,
        num_layers=num_layers,
        patch_size=patch_size,
        num_heads=num_heads,
        embed_dim=hidden_dim,
        use_patch_bias=True,
        use_pre_norm=False,
        attn_key_dim=0,
        attn_qv_bias=True,
        attn_qkv_bias=attn_qkv_bias,
        attn_return_weight=True,
        attn_return_bias=True,
        attn_layer_scale=0.0,
        attn_dropout=0,
        use_abs_pos_emb=True,
        use_abs_pos_emb_on_cls_token=True,
        use_rot_pos_emb=False,
        mlp_ratio=mlp_ratio,
        use_gated_mlp=False,
        use_mlp_norm=False,
        use_mean_pooling_head=True,
        use_cat_head=False,
        max_block_size=77,
        text_positional_dropout=0,
        text_use_positional_embedding=True,
        inputs=inputs,
        include_head=include_head,
        num_classes=num_classes,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )

    model_name = "EVA"
    if num_layers == 12:
        if num_heads < 5:
            model_name += "-tiny"
        elif num_heads < 8:
            model_name += "-small"
        else:
            model_name += "-base"
    elif num_layers == 24:
        model_name += "-large"
    elif num_layers == 32:
        model_name += "-huge"
    elif num_layers == 40:
        model_name += "-gaint"
    model_name += f"-{patch_size}"

    model = Model(inputs=inputs, outputs=backbone.outputs, name=model_name)
    return model


def EVA_T14(
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

    model = EVA(
        num_layers=12,
        patch_size=14,
        num_heads=3,
        hidden_dim=192,
        mlp_ratio=4,
        attn_qkv_bias=True,
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


def EVA_S14(
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

    model = EVA(
        num_layers=12,
        patch_size=14,
        num_heads=6,
        hidden_dim=384,
        mlp_ratio=4,
        attn_qkv_bias=True,
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


def EVA_B14(
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

    model = EVA(
        num_layers=12,
        patch_size=14,
        num_heads=12,
        hidden_dim=768,
        mlp_ratio=4,
        attn_qkv_bias=True,
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


def EVA_L14(
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

    model = EVA(
        num_layers=24,
        patch_size=14,
        num_heads=16,
        hidden_dim=1024,
        mlp_ratio=4,
        attn_qkv_bias=True,
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


def EVA_H14(
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

    model = EVA(
        num_layers=32,
        patch_size=14,
        num_heads=16,
        hidden_dim=1280,
        mlp_ratio=4,
        attn_qkv_bias=True,
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


def EVA_G14(
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

    model = EVA(
        num_layers=40,
        patch_size=14,
        num_heads=16,
        hidden_dim=1408,
        mlp_ratio=6144 / 1408,
        attn_qkv_bias=False,
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
