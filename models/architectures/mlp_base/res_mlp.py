"""
  # Description:
    - The following table comparing the params of the gMLP in Tensorflow on 
    size 224 x 224 x 3:
      
       ---------------------------------------------
      |        Model Name         |    Params       |
      |---------------------------------------------|
      |      ResMLP-small-12      |    15,350,872   |
      |---------------------------------------------|
      |      ResMLP-small-24      |    30,020,680   |
      |---------------------------------------------|
      |      ResMLP-small-36      |    44,690,488   |
      |---------------------------------------------|
      |      ResMLP-base-24       |   115,736,776   |
       ---------------------------------------------
       
  # Reference:
    - [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)
    - Source: https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/res_mlp.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import ChannelAffine, SAMModel, get_activation_from_name
from utils.model_processing import _obtain_input_shape



def res_mlp_block(inputs, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = ChannelAffine(use_bias=True, axis=-1, name=name + "norm_1")(inputs)
    nn = Permute((2, 1), name=name + "permute_1")(nn)
    nn = Dense(nn.shape[-1], name=name + "token_mixing")(nn)
    nn = Permute((2, 1), name=name + "permute_2")(nn)
    nn = ChannelAffine(use_bias=False, axis=-1, name=name + "gamma_1")(nn)
    
    if drop_rate > 0:
        nn = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = add([inputs, nn])

    nn = ChannelAffine(use_bias=True, axis=-1, name=name + "norm_2")(token_out)
    nn = Dense(channels_mlp_dim, name=name + "channel_mixing_1")(nn)
    nn = get_activation_from_name(activation, name=name + activation)(nn)
    nn = Dense(inputs.shape[-1], name=name + "channel_mixing_2")(nn)
    channel_out = ChannelAffine(use_bias=False, axis=-1, name=name + "gamma_2")(nn)
    
    if drop_rate > 0:
        channel_out = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
        
    nn = add([channel_out, token_out])
    return nn

    
def ResMLP(stem_width,
           patch_size,
           num_blocks,
           channels_mlp_dim,
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           final_activation="softmax",
           classes=1000,
           sam_rho=0.0,
           drop_rate=0.,
           drop_connect_rate=0.):
        
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
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

    x = Conv2D(filters=stem_width, 
               kernel_size=patch_size, 
               strides=patch_size, 
               padding="valid", name="stem")(img_input)
    x = Reshape(target_shape=(-1, stem_width))(x)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_rate, (list, tuple)) else [drop_rate, drop_rate]
    
    for ii in range(num_blocks):
        name = "{}_{}_".format("res_block", str(ii + 1))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        x = res_mlp_block(x, channels_mlp_dim=channels_mlp_dim, drop_rate=block_drop_rate, activation='gelu', name=name)
        
    x = ChannelAffine(axis=-1, name="pre_head_norm")(x)
    
    if include_top:
        x = GlobalAveragePooling1D()(x)
        x = Dropout(rate=drop_rate)(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

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
    if stem_width == 384:
        model = __build_model(inputs, x, sam_rho, name=f'ResMLP-S{num_blocks}')
    elif stem_width == 768:
        model = __build_model(inputs, x, sam_rho, name=f'ResMLP-B{num_blocks}')
    else:
        model = __build_model(inputs, x, sam_rho, name=f'ResMLP-{num_blocks}')

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


def ResMLP_S12(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               drop_rate=0.1,
               drop_connect_rate=0.1) -> Model:
    
    model = ResMLP(stem_width=384,
                   patch_size=16,
                   num_blocks=12,
                   channels_mlp_dim=384 * 4,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes,
                   sam_rho=sam_rho,
                   drop_rate=drop_rate,
                   drop_connect_rate=drop_connect_rate)
    return model


def ResMLP_S24(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               drop_rate=0.1,
               drop_connect_rate=0.1) -> Model:
        
    model = ResMLP(stem_width=384,
                   patch_size=16,
                   num_blocks=24,
                   channels_mlp_dim=384 * 4,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes,
                   sam_rho=sam_rho,
                   drop_rate=drop_rate,
                   drop_connect_rate=drop_connect_rate)
    return model


def ResMLP_S36(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               drop_rate=0.1,
               drop_connect_rate=0.1) -> Model:
    
    model = ResMLP(stem_width=384,
                   patch_size=16,
                   num_blocks=36,
                   channels_mlp_dim=384 * 4,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes,
                   sam_rho=sam_rho,
                   drop_rate=drop_rate,
                   drop_connect_rate=drop_connect_rate)
    return model


def ResMLP_B24(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               drop_rate=0.1,
               drop_connect_rate=0.1) -> Model:
    
    model = ResMLP(stem_width=768,
                   patch_size=8,
                   num_blocks=24,
                   channels_mlp_dim=768 * 4,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes,
                   sam_rho=sam_rho,
                   drop_rate=drop_rate,
                   drop_connect_rate=drop_connect_rate)
    return model