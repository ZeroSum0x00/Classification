from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from .beit import BEiT
from models.layers import get_activation_from_name
from utils.model_processing import _obtain_input_shape, drop_connect_rates_split


def ViT_Text(vocab_size,
             max_block_size,
             num_layers,
             patch_size,
             num_heads,
             hidden_dim,
             text_use_positional_embedding=True,
             include_top=True, 
             weights='imagenet',
             input_tensor=None, 
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000,
             sam_rho=0.0,
             norm_eps=1e-6,
             text_positional_dropout=0,
             drop_rate=0.1):

    backbone = BEiT(vocab_size=vocab_size,
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
                    include_top=False,
                    weights=None,
                    pooling=pooling,
                    activation='gelu',
                    norm_eps=norm_eps,
                    drop_rate=drop_rate)
                 
    i = backbone.input
    x = backbone.output

    if include_top:
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avgpool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_maxpool')(x)

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, x, name=name + '_SAM')
        else:
            return Model(inputs, x, name=name)
            
    # Create model.
    if num_layers == 12:
        if num_heads < 5:
            model = __build_model(i, x, sam_rho, name=f'ViT-Text-Tiny-{patch_size}')
        else:
            model = __build_model(i, x, sam_rho, name=f'ViT-Text-Base-{patch_size}')
    elif num_layers == 24:
        model = __build_model(i, x, sam_rho, name=f'ViT-Text-Large-{patch_size}')
    elif num_layers == 32:
        model = __build_model(i, x, sam_rho, name=f'ViT-Text-Huge-{patch_size}')
    else:
        model = __build_model(i, x, sam_rho, name=f'ViT-Text-{patch_size}')

    if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
    return model


def ViT_Text_Large14(vocab_size=49408,
                     max_block_size=77,
                     include_top=True, 
                     weights='imagenet',
                     input_tensor=None, 
                     input_shape=None,
                     pooling=None,
                     final_activation="softmax",
                     classes=1000,
                     sam_rho=0.0,
                     norm_eps=1e-6,
                     drop_rate=0.1):

    model = ViT_Text(num_layers=12,
                     patch_size=14,
                     num_heads=12,
                     hidden_dim=768,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor,
                     input_shape=input_shape,
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model