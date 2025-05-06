"""
  # Description:
    - The following table comparing the params of the EVA-02 base BEiT block
    in Tensorflow on in Tensorflow on size 336 x 336 x 3:

       -------------------------------------------
      |       Model Name      |      Params       |
      |-------------------------------------------|
      |     EVA02-Tiny-14     |      5,645,800    |
      |-----------------------|-------------------|
      |     EVA02-Small-14    |     21,907,432    |
      |-----------------------|-------------------|
      |     EVA02-Base-14     |     86,330,344    |
      |-----------------------|-------------------|
      |     EVA02-Large-14    |    304,030,632    |
      |-----------------------|-------------------|
      |     EVA02-Huge-14     |    631,907,944    |
      |-----------------------|-------------------|
      |     EVA02-Gaint-14    |    954,763,816    |
       -------------------------------------------

  # Reference:
    - [EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/pdf/2303.11331.pdf)

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



def EVA02(
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    use_mlp_norm=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
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
            model = __build_model(inputs, x, sam_rho, name=f"EVA02-Tiny-{patch_size}")
        elif num_heads < 8:
            model = __build_model(inputs, x, sam_rho, name=f"EVA02-Small-{patch_size}")
        else:
            model = __build_model(inputs, x, sam_rho, name=f"EVA02-Base-{patch_size}")
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f"EVA02-Large-{patch_size}")
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f"EVA02-Huge-{patch_size}")
    elif num_layers == 40:
        model = __build_model(inputs, x, sam_rho, name=f"EVA02-Gaint-{patch_size}")
    else:
        model = __build_model(inputs, x, sam_rho, name=f"EVA02-{patch_size}")
        
    return model


def EVA02_T14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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


def EVA02_S14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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


def EVA02_B14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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

                     
def EVA02_L14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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


def EVA02_H14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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


def EVA02_G14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    sam_rho=0.0,
    norm_eps=1e-6,
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
