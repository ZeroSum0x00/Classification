"""
    Overview:
        This file summarizes parameter configurations and key details for 
        the Meta-Transformer architecture: a **unified Transformer framework** 
        for **multimodal learning** (vision, language, audio, video, etc.).
    
        Unlike conventional modality-specific models, Meta-Transformer enables 
        a shared representation space and architectural backbone, unifying multiple modalities 
        via a modality-agnostic encoder-decoder structure.
    
    Key Features:
        - Unified architecture for vision, language, audio, video, and multimodal tasks.
        - Works across tasks: classification, captioning, retrieval, QA, VQA, etc.
        - Modularity: Plug-and-play encoders/decoders for different modalities.
        - Uses a shared latent representation for all inputs and outputs.
        - Efficient training and inference across tasks and domains.
    
    Model Parameter Comparison:
         --------------------------------------------------
        |           Model Name           |      Params     |
        |--------------------------------+-----------------|
        |     MetaTransformer-tiny-14    |     5,645,224   |
        |--------------------------------+-----------------|
        |     MetaTransformer-tiny-16    |     5,679,784   |
        |--------------------------------+-----------------|
        |     MetaTransformer-small-14   |    21,906,280   |
        |--------------------------------+-----------------|
        |     MetaTransformer-small-16   |    21,975,400   |
        |--------------------------------+-----------------|
        |     MetaTransformer-base-14    |    86,278,888   |
        |--------------------------------+-----------------|
        |     MetaTransformer-base-16    |    86,417,128   |
        |--------------------------------+-----------------|
        |     MetaTransformer-large-14   |   303,941,608   |
        |--------------------------------+-----------------|
        |     MetaTransformer-large-16   |   451,164,232   |
        |--------------------------------+-----------------|
        |     MetaTransformer-huge-14    |   599,870,344   |
        |--------------------------------+-----------------|
        |     MetaTransformer-huge-16    |   909,332,968   |
        |--------------------------------+-----------------|
        |     MetaTransformer-gaint-14   |       -         |
        |--------------------------------+-----------------|
        |     MetaTransformer-gaint-16   |       -         |
         --------------------------------------------------

    References:
        - Paper: "Meta-Transformer: A Unified Framework for Multimodal Learning"
          https://arxiv.org/pdf/2307.10802
          
        - PyTorch repository (not official):
          https://github.com/OpenGVLab/MetaTransformer (community implementation)
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/meta_transformer.py
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from .beit import BEiT
from models.layers import get_activation_from_name
from utils.model_processing import process_model_input, check_regularizer



def MetaTransformer(
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

    if weights == 'imagenet' and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    regularizer_decay = check_regularizer(regularizer_decay)
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224 if num_layers < 24 else 336,
        min_size=32,
        weights=weights
    )
    
    backbone = BEiT(
        vocab_size=0,
        num_layers=num_layers,
        patch_size=patch_size,
        num_heads=num_heads,
        embed_dim=hidden_dim,
        use_patch_bias=False,
        use_pre_norm=True,
        attn_key_dim=0,
        attn_qv_bias=False,
        attn_qkv_bias=True,
        attn_return_weight=True,
        attn_return_bias=True,
        attn_layer_scale=0.0,
        attn_dropout=0,
        use_abs_pos_emb=True,
        use_abs_pos_emb_on_cls_token=True,
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

    model_name = "MetaTransformer"
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


def MetaTransformer_T14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=12,
        patch_size=14,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_T16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_S14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=12,
        patch_size=14,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_S16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_B14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=12,
        patch_size=14,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_B16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_L14(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=24,
        patch_size=14,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_L16(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=24,
        patch_size=16,
        num_heads=24,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_H14(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=32,
        patch_size=14,
        num_heads=18,
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
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_H16(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=32,
        patch_size=16,
        num_heads=24,
        hidden_dim=1536,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_G14(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=40,
        patch_size=14,
        num_heads=24,
        hidden_dim=1536,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def MetaTransformer_G16(
    inputs=[336, 336, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = MetaTransformer(
        num_layers=40,
        patch_size=16,
        num_heads=32,
        hidden_dim=2048,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model
