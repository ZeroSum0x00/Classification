import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from .beit import BEiT
from models.layers import get_activation_from_name, SAMModel
from utils.model_processing import process_model_input



def ViT_Text(
    vocab_size,
    max_block_size,
    num_layers,
    patch_size,
    num_heads,
    hidden_dim,
    text_positional_dropout=0,
    text_use_positional_embedding=True,
    inputs=[None],
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

    backbone = BEiT(
        vocab_size=vocab_size,
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
        max_block_size=max_block_size,
        text_positional_dropout=text_positional_dropout,
        text_use_positional_embedding=text_use_positional_embedding,
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
                 
    i = backbone.input
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
            
    # Create model.
    if num_layers == 12:
        if num_heads < 5:
            model = __build_model(i, x, sam_rho, name=f"ViT-Text-Tiny-{patch_size}")
        else:
            model = __build_model(i, x, sam_rho, name=f"ViT-Text-Base-{patch_size}")
    elif num_layers == 24:
        model = __build_model(i, x, sam_rho, name=f"ViT-Text-Large-{patch_size}")
    elif num_layers == 32:
        model = __build_model(i, x, sam_rho, name=f"ViT-Text-Huge-{patch_size}")
    else:
        model = __build_model(i, x, sam_rho, name=f"ViT-Text-{patch_size}")
        
    return model


def ViT_Text_L14(
    inputs=[None],
    vocab_size=49408,
    max_block_size=77,
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

    model = ViT_Text(
        vocab_size=vocab_size,
        max_block_size=max_block_size,
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
