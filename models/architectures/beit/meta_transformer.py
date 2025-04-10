"""
  # Description:
    - The following table comparing the params of the Explore the limits of Meta-Transformer, base BEiT block
    in Tensorflow on in Tensorflow on size 224 x 224 x 3 for 'Base' variant and 336 x 336 x 3 for 'Large' variant:

       --------------------------------------------------
      |           Model Name           |      Params     |
      |--------------------------------------------------|
      |     MetaTransformer-Tiny-14    |     5,645,224   |
      |--------------------------------------------------|
      |     MetaTransformer-Tiny-16    |     5,679,784   |
      |--------------------------------------------------|
      |     MetaTransformer-Small-14   |    21,906,280   |
      |--------------------------------------------------|
      |     MetaTransformer-Small-16   |    21,975,400   |
      |--------------------------------------------------|
      |     MetaTransformer-Base-14    |    86,278,888   |
      |--------------------------------------------------|
      |     MetaTransformer-Base-16    |    86,417,128   |
      |--------------------------------------------------|
      |     MetaTransformer-Large-14   |   303,941,608   |
      |--------------------------------------------------|
      |     MetaTransformer-Large-16   |   451,164,232   |
      |--------------------------------------------------|
      |     MetaTransformer-Huge-14    |   599,870,344   |
      |--------------------------------------------------|
      |     MetaTransformer-Huge-16    |   909,332,968   |
      |--------------------------------------------------|
      |     MetaTransformer-Gaint-14   |       -         |
      |--------------------------------------------------|
      |     MetaTransformer-Gaint-16   |       -         |
       --------------------------------------------------

  # Reference:
    - [Meta-Transformer: A Unified Framework for Multimodal Learning](https://arxiv.org/pdf/2307.10802.pdf)

"""

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
from models.layers import get_activation_from_name, SAMModel
from utils.model_processing import _obtain_input_shape, drop_connect_rates_split


def MetaTransformer(num_layers,
                    patch_size,
                    num_heads,
                    hidden_dim,
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
                 
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224 if num_layers < 24 else 336,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    backbone = BEiT(vocab_size=0,
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
                    include_top=False,
                    weights=None,
                    input_tensor=img_input, 
                    pooling=pooling,
                    activation='gelu',
                    norm_eps=norm_eps,
                    drop_rate=drop_rate)
    x = backbone.output

    if include_top:
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avgpool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_maxpool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, x, name=name + '_SAM')
        else:
            return Model(inputs, x, name=name)
            
    # Create model.
    if num_layers == 12:
        if num_heads < 5:
            model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Tiny-{patch_size}')
        elif num_heads < 8:
            model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Small-{patch_size}')
        else:
            model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Base-{patch_size}')
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Large-{patch_size}')
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Huge-{patch_size}')
    elif num_layers == 40:
        model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-Gaint-{patch_size}')
    else:
        model = __build_model(inputs, x, sam_rho, name=f'Meta-Transformer-{patch_size}')

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


def MetaTransformer_T14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
                            patch_size=14,
                            num_heads=3,
                            hidden_dim=192,
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


def MetaTransformer_T16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
                            patch_size=16,
                            num_heads=3,
                            hidden_dim=192,
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


def MetaTransformer_S14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
                            patch_size=14,
                            num_heads=6,
                            hidden_dim=384,
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


def MetaTransformer_S16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
                            patch_size=16,
                            num_heads=6,
                            hidden_dim=384,
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


def MetaTransformer_B14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
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


def MetaTransformer_B16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=12,
                            patch_size=16,
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


def MetaTransformer_L14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=24,
                            patch_size=14,
                            num_heads=16,
                            hidden_dim=1024,
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


def MetaTransformer_L16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=24,
                            patch_size=16,
                            num_heads=24,
                            hidden_dim=1248,
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


def MetaTransformer_H14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=32,
                            patch_size=14,
                            num_heads=18,
                            hidden_dim=1248,
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


def MetaTransformer_H16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=32,
                            patch_size=16,
                            num_heads=24,
                            hidden_dim=1536,
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


def MetaTransformer_G14(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=40,
                            patch_size=14,
                            num_heads=24,
                            hidden_dim=1536,
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


def MetaTransformer_G16(include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation="softmax",
                        classes=1000,
                        sam_rho=0.0,
                        norm_eps=1e-6,
                        drop_rate=0.1):

    model = MetaTransformer(num_layers=40,
                            patch_size=16,
                            num_heads=32,
                            hidden_dim=2048,
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