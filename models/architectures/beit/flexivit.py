"""
    Overview:
        This script summarizes the parameter counts for the FlexiViT model family, 
        a scalable Vision Transformer architecture designed for **resolution-agnostic inference**.
        FlexiViT models are trained once and can be deployed across multiple input sizes
        (e.g., 224x224, 384x384, 512x512, etc.) without needing finetuning.
    
        The table below shows parameter statistics based on TensorFlow-converted models 
        using patch size 16.
    
    Model Parameter Comparison:
         -------------------------------------------
        |        Model Name       |      Params     |
        |-------------------------+-----------------|
        |     FlexiViT-Tiny-16    |     5,679,592   |
        |-------------------------+-----------------|
        |     FlexiViT-Small-16   |    21,975,016   |
        |-------------------------+-----------------|
        |     FlexiViT-Base-16    |    86,416,360   |
        |-------------------------+-----------------|
        |     FlexiViT-Large-16   |   304,124,904   |
        |-------------------------+-----------------|
        |     FlexiViT-Huge-16    |   600,813,160   |
        |-------------------------+-----------------|
        |     FlexiViT-Gaint-16   |   954,810,856   |
         -------------------------------------------

    Notes:
        - All models support **multi-resolution inference** without retraining.
        - FlexiViT models extend standard ViTs using position interpolation and relative attention.
        - Parameter counts are approximate; actual values may vary slightly by implementation.
        - Patch size is 16x16 for all variants.
        - Trained using ImageNet-21k and other large-scale datasets.
    
    References:
        - Paper: "FlexiViT: One Model for All Patch Sizes"
          https://arxiv.org/abs/2212.08013
          
        - Official PyTorch repository:
          https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/flexivit.py
    """

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from .beit import BEiT
from models.layers import get_activation_from_name
from utils.model_processing import process_model_input, check_regularizer



def FlexiViT(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
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
        attn_qv_bias=False,
        attn_qkv_bias=True,
        attn_return_weight=True,
        attn_return_bias=True,
        attn_layer_scale=0.0,
        attn_dropout=0,
        use_abs_pos_emb=True,
        use_abs_pos_emb_on_cls_token=False,
        use_rot_pos_emb=False,
        mlp_ratio=4,
        use_gated_mlp=False,
        use_mlp_norm=False,
        use_mean_pooling_head=False,
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

    model_name = "FlexiViT"
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


def FlexiViT_T16(
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

    model = FlexiViT(
        num_layers=12,
        patch_size=16,
        num_heads=3,
        hidden_dim=192,
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


def FlexiViT_S16(
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

    model = FlexiViT(
        num_layers=12,
        patch_size=16,
        num_heads=6,
        hidden_dim=384,
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


def FlexiViT_B16(
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

    model = FlexiViT(
        num_layers=12,
        patch_size=16,
        num_heads=12,
        hidden_dim=768,
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


def FlexiViT_L16(
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

    model = FlexiViT(
        num_layers=24,
        patch_size=16,
        num_heads=16,
        hidden_dim=1024,
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


def FlexiViT_H16(
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

    model = FlexiViT(
        num_layers=32,
        patch_size=16,
        num_heads=16,
        hidden_dim=1248,
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


def FlexiViT_G16(
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

    model = FlexiViT(
        num_layers=40,
        patch_size=16,
        num_heads=16,
        hidden_dim=1408,
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
