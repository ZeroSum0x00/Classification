"""
  # Description:
    - The following table comparing the params of the self-DIstillation with NO labels version 2 (DINO v2) base BEiT block
    in Tensorflow on in Tensorflow on size 518 x 518 x 3:

       ---------------------------------------------
      |       Model Name        |      Params       |
      |---------------------------------------------|
      |     DINOv2-Tiny-14      |      5,841,640    |
      |---------------------------------------------|
      |     DINOv2-Small-14     |     22,299,112    |
      |-------------------------|-------------------|
      |     DINOv2-Base-14      |     87,064,552    |
      |-------------------------|-------------------|
      |     DINOv2-Large-14     |    305,013,736    |
      |-------------------------|-------------------|
      |     DINOv2-Huge-14      |    601,196,968    |
      |-------------------------|-------------------|
      |     DINOv2-Gaint-14     |  1,137,447,912    |
       ---------------------------------------------

  # Reference:
    - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf)

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from .beit import BEiT
from models.layers import get_activation_from_name, SAMModel
from utils.model_processing import process_model_input



def DINOv2(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    mlp_ratio,
    use_gated_mlp=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
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
        attn_layer_scale=1.0,
        attn_dropout=0,
        use_abs_pos_emb=True,
        use_abs_pos_emb_on_cls_token=True,
        use_rot_pos_emb=False,
        mlp_ratio=mlp_ratio,
        use_gated_mlp=use_gated_mlp,
        use_mlp_norm=False,
        use_mean_pooling_head=False,
        use_cat_head=True,
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
        x = Sequential([
            Dropout(drop_rate),
            Dense(
                units=1 if num_classes == 2 else num_classes,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l2(regularizer_decay),
            ),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
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
            model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Tiny-{patch_size}")
        elif num_heads < 8:
            model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Small-{patch_size}")
        else:
            model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Base-{patch_size}")
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Large-{patch_size}")
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Huge-{patch_size}")
    elif num_layers == 40:
        model = __build_model(inputs, x, sam_rho, name=f"DINOv2-Gaint-{patch_size}")
    else:
        model = __build_model(inputs, x, sam_rho, name=f"DINOv2-{patch_size}")

    return model


def DINOv2_T14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=12,
        patch_size=14,
        num_heads=3,
        hidden_dim=192,
        mlp_ratio=4,
        use_gated_mlp=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DINOv2_S14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=12,
        patch_size=14,
        num_heads=6,
        hidden_dim=384,
        mlp_ratio=4,
        use_gated_mlp=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model

                        
def DINOv2_B14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=12,
        patch_size=14,
        num_heads=12,
        hidden_dim=768,
        mlp_ratio=4,
        use_gated_mlp=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DINOv2_L14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=24,
        patch_size=14,
        num_heads=16,
        hidden_dim=1024,
        mlp_ratio=4,
        use_gated_mlp=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DINOv2_H14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=32,
        patch_size=14,
        num_heads=18,
        hidden_dim=1248,
        mlp_ratio=4,
        use_gated_mlp=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DINOv2_G14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = DINOv2(
        num_layers=40,
        patch_size=14,
        num_heads=24,
        hidden_dim=1536,
        mlp_ratio=4096 / 1536,
        use_gated_mlp=True,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model
