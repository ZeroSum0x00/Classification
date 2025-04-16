"""
  # Description:
    - The following table comparing the params of the Explore the limits of Flexible Vision Transformer (FlexiViT) base BEiT block
    in Tensorflow on in Tensorflow on size 240 x 240 x 3:

       -------------------------------------------
      |        Model Name       |      Params     |
      |-------------------------------------------|
      |     FlexiViT-Tiny-16    |     5,679,592   |
      |-------------------------------------------|
      |     FlexiViT-Small-16   |    21,975,016   |
      |-------------------------|-----------------|
      |     FlexiViT-Base-16    |    86,416,360   |
      |-------------------------|-----------------|
      |     FlexiViT-Large-16   |   304,124,904   |
      |-------------------------------------------|
      |     FlexiViT-Huge-16    |   600,813,160   |
      |-------------------------------------------|
      |     FlexiViT-Gaint-16   |   954,810,856   |
       -------------------------------------------

  # Reference:
    - [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf)

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from .beit import BEiT
from models.layers import get_activation_from_name, SAMModel
from utils.model_processing import process_model_input



def FlexiViT(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    inputs=[224, 224, 3],
    include_head=True, 
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
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
        include_head=False,
        weights=None,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    
    x = backbone.output

    if include_head:
        x = Dense(
            units=1 if num_classes == 2 else num_classes, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name="predictions"
        )(x)
        
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="global_avgpool")(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D(name="global_maxpool")(x)

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, outputs, name=name + "_SAM")
        else:
            return Model(inputs=inputs, outputs=outputs, name=name)
            
    if num_layers == 12:
        if num_heads < 5:
            model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Tiny-{patch_size}")
        elif num_heads < 8:
            model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Small-{patch_size}")
        else:
            model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Base-{patch_size}")
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Large-{patch_size}")
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Huge-{patch_size}")
    elif num_layers == 40:
        model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-Gaint-{patch_size}")
    else:
        model = __build_model(inputs, x, sam_rho, name=f"FlexiViT-{patch_size}")
        
    return model


def FlexiViT_T16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def FlexiViT_S16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def FlexiViT_B16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def FlexiViT_L16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def FlexiViT_H16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def FlexiViT_G16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
        pooling=pooling, 
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model