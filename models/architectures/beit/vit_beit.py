"""
  # Description:
    - The following table comparing the params of the Vision Transformer (ViT) base BEiT block
    in Tensorflow on in Tensorflow on size 224 x 224 x 3:

       -------------------------------------------
      |       Model Name       |     Params       |
      |-------------------------------------------|
      |    ViT-BEiT-Tiny-14    |     5,645,032    |
      |-------------------------------------------|
      |    ViT-BEiT-Tiny-16    |     5,679,592    |
      |-------------------------------------------|
      |    ViT-BEiT-Tiny-32    |     6,121,960    |
      |-------------------------------------------|
      |    ViT-BEiT-Small-14   |    21,905,896    |
      |-------------------------------------------|
      |    ViT-BEiT-Small-16   |    21,975,016    |
      |-------------------------------------------|
      |    ViT-BEiT-Small-32   |    22,859,752    |
      |-------------------------------------------|
      |    ViT-BEiT-Base-14    |    86,278,120    |
      |-------------------------------------------|
      |    ViT-BEiT-Base-16    |    86,416,360    |
      |-------------------------------------------|
      |    ViT-BEiT-Base-32    |    88,185,832    |
      |-------------------------------------------|
      |    ViT-BEiT-Large-14   |   303,940,584    |
      |-------------------------------------------|
      |    ViT-BEiT-Large-16   |   304,124,904    |
      |-------------------------------------------|
      |    ViT-BEiT-Large-32   |   306,484,200    |
      |-------------------------------------------|
      |    ViT-BEiT-Huge-14    |   631,716,840    |
      |-------------------------------------------|
      |    ViT-BEiT-Huge-16    |   631,947,240    |
      |-------------------------------------------|
      |    ViT-BEiT-Huge-32    |   634,896,360    |
      |-------------------------------------------|
      |    ViT-BEiT-Gaint-14   |  1,135,707,112   |
      |-------------------------------------------|
      |    ViT-BEiT-Gaint-16   |        -         |
      |-------------------------------------------|
      |    ViT-BEiT-Gaint-16   |        -         |
       -------------------------------------------

  # Reference:
    - [An image is worth 16x16 words: transformers for image recognition 
       at scale](https://arxiv.org/pdf/2010.11929.pdf)
    - [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254v2.pdf)

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



def ViT_BEiT(
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
            model = __build_model(inputs, x, sam_rho, name=f"ViT-BEiT-Tiny-{patch_size}")
        else:
            model = __build_model(inputs, x, sam_rho, name=f"ViT-BEiT-Base-{patch_size}")
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f"ViT-BEiT-Large-{patch_size}")
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f"ViT-BEiT-Huge-{patch_size}")
    else:
        model = __build_model(inputs, x, sam_rho, name=f"ViT-BEiT-{patch_size}")
        
    return model


def ViT_BEiT_T14(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=14,
        num_heads=3,
        hidden_dim=192,
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


def ViT_BEiT_T16(
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

    model = ViT_BEiT(
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
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def ViT_BEiT_T32(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=32,
        num_heads=3,
        hidden_dim=192,
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


def ViT_BEiT_S14(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=14,
        num_heads=6,
        hidden_dim=384,
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


def ViT_BEiT_S16(
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

    model = ViT_BEiT(
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
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def ViT_BEiT_S32(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=32,
        num_heads=6,
        hidden_dim=384,
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


def ViT_BEiT_B14(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=14,
        num_heads=12,
        hidden_dim=768,
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


def ViT_BEiT_B16(
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

    model = ViT_BEiT(
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
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def ViT_BEiT_B32(
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

    model = ViT_BEiT(
        num_layers=12,
        patch_size=32,
        num_heads=12,
        hidden_dim=768,
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


def ViT_BEiT_L14(
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

    model = ViT_BEiT(
        num_layers=24,
        patch_size=14,
        num_heads=16,
        hidden_dim=1024,
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

            
def ViT_BEiT_L16(
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

    model = ViT_BEiT(
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
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        sam_rho=sam_rho,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def ViT_BEiT_L32(
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

    model = ViT_BEiT(
        num_layers=24,
        patch_size=32,
        num_heads=16,
        hidden_dim=1024,
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


def ViT_BEiT_H14(
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

    model = ViT_BEiT(
        num_layers=32,
        patch_size=14,
        num_heads=16,
        hidden_dim=1280,
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

           
def ViT_BEiT_H16(
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

    model = ViT_BEiT(
        num_layers=32,
        patch_size=16,
        num_heads=16,
        hidden_dim=1280,
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


def ViT_BEiT_H32(
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

    model = ViT_BEiT(
        num_layers=32,
        patch_size=32,
        num_heads=16,
        hidden_dim=1280,
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


def ViT_BEiT_G14(
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

    model = ViT_BEiT(
        num_layers=40,
        patch_size=14,
        num_heads=16,
        hidden_dim=1536,
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


def ViT_BEiT_G16(
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

    model = ViT_BEiT(
        num_layers=40,
        patch_size=16,
        num_heads=16,
        hidden_dim=1536,
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


def ViT_BEiT_G32(
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

    model = ViT_BEiT(
        num_layers=40,
        patch_size=32,
        num_heads=16,
        hidden_dim=1536,
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
