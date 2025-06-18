"""
    Overview:
        This script summarizes parameter counts for the EVA-02 model family (a.k.a. EVA-CLIP),
        which builds upon BEiT-style Vision Transformers and is pretrained using large-scale contrastive learning
        (CLIP-style) with a focus on visual representation quality and scalability.
    
        EVA-02 has been adopted as the visual backbone in models like Segment Anything (SAM) and InternVL.
    
        All models are implemented in TensorFlow, using input resolution 518 x 518 x 3 and patch size 14.
        
    Model Parameter Comparison:
         -------------------------------------------
        |       Model Name      |      Params       |
        |-----------------------+-------------------|
        |     EVA02-Tiny-14     |      5,645,800    |
        |-----------------------+-------------------|
        |     EVA02-Small-14    |     21,907,432    |
        |-----------------------+-------------------|
        |     EVA02-Base-14     |     86,330,344    |
        |-----------------------+-------------------|
        |     EVA02-Large-14    |    304,030,632    |
        |-----------------------+-------------------|
        |     EVA02-Huge-14     |    631,907,944    |
        |-----------------------+-------------------|
        |     EVA02-Gaint-14    |    954,763,816    |
         -------------------------------------------
       
    Notes:
        - Trained with contrastive vision-language objectives (CLIP-style).
        - "-14" indicates a patch size of 14×14 in the ViT backbone.
        - Parameter count is for the image encoder only.
    
    References:
        - Paper: "EVA-02: A Visual Representation for Neon Genesis"
          https://arxiv.org/pdf/2303.11331.pdf
          
        - Official PyTorch repository:
          https://github.com/baaivision/EVA/tree/master/EVA-02
          
        - TensorFlow/Keras port by leondgarse:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/eva02.py
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout

from .beit import BEiT
from models.layers import get_activation_from_name
from utils.model_processing import process_model_input, check_regularizer



def EVA02(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    use_mlp_norm=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
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
        attn_qkv_bias=False,
        attn_return_weight=True,
        attn_return_bias=True,
        attn_layer_scale=0.0,
        attn_dropout=0,
        use_abs_pos_emb=True,
        use_abs_pos_emb_on_cls_token=True,
        use_rot_pos_emb=False,
        mlp_ratio=4 * 2 / 3,
        use_gated_mlp=True,
        use_mlp_norm=use_mlp_norm,
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

    model_name = "EVA02"
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


def EVA02_T14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=12,
        patch_size=14,
        num_heads=3,
        hidden_dim=192,
        use_mlp_norm=False,
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


def EVA02_S14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=12,
        patch_size=14,
        num_heads=6,
        hidden_dim=384,
        use_mlp_norm=False,
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


def EVA02_B14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=12,
        patch_size=14,
        num_heads=12,
        hidden_dim=768,
        use_mlp_norm=True,
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

                     
def EVA02_L14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=24,
        patch_size=14,
        num_heads=16,
        hidden_dim=1024,
        use_mlp_norm=True,
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


def EVA02_H14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=32,
        patch_size=14,
        num_heads=16,
        hidden_dim=1280,
        use_mlp_norm=True,
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


def EVA02_G14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.1,
    drop_rate=0.1
):

    model = EVA02(
        num_layers=40,
        patch_size=14,
        num_heads=16,
        hidden_dim=1408,
        use_mlp_norm=True,
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
