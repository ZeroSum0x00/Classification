"""
  # Description:
    - The following table comparing the params of the Inception v3 in Tensorflow on 
    size 299 x 299 x 3:

       --------------------------------------
      |     Model Name     |    Params       |
      |--------------------------------------|
      |    Inception v3    |   23,869,000    |
       --------------------------------------

  # Reference:
    - [An image is worth 16x16 words: transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf)
    - Source: https://github.com/faustomorales/vit-keras

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def convolution_block(inputs, 
                      filters, 
                      kernel_size, 
                      strides=(1, 1), 
                      padding='same', 
                      use_bias=True, 
                      activation="relu", 
                      normalizer='batch-norm', 
                      name=None):
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               strides=strides, 
               padding=padding, 
               use_bias=use_bias,
               name=name + '_conv')(inputs)
    x = get_normalizer_from_name(normalizer, name=name + '_bn')(x)
    x = get_activation_from_name(activation, name=name + '_activ')(x)
    return x

    
def Inception_v3(include_top=True, 
                 weights='imagenet',
                 input_tensor=None, 
                 input_shape=None,
                 pooling=None,
                 final_activation='softmax',
                 classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=75,
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # stem
    x = convolution_block(img_input, 32, (3, 3), (2, 2), padding='valid', use_bias=False, name='stem1')
    x = convolution_block(x, 32, (3, 3), padding='valid', use_bias=False, name='stem2')
    x = convolution_block(x, 64, (3, 3), use_bias=False, name='stem3')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='stem_pooling1')(x)

    x = convolution_block(x, 80, (1, 1), padding='valid', use_bias=False, name='stem4')
    x = convolution_block(x, 192, (3, 3), padding='valid', use_bias=False, name='stem5')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='stem_pooling2')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed0_b11')
                     
    branch5x5 = convolution_block(x, 48, (1, 1), use_bias=False, name='mixed0_b21')
    branch5x5 = convolution_block(branch5x5, 64, (5, 5), use_bias=False, name='mixed0_b22')

    branch3x3dbl = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed0_b31')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed0_b32')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed0_b33')

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='mixed0_b41')(x)
    branch_pool = convolution_block(branch_pool, 32, (1, 1), use_bias=False, name='mixed0_b42')
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed1_b11')

    branch5x5 = convolution_block(x, 48, (1, 1), use_bias=False, name='mixed1_b21')
    branch5x5 = convolution_block(branch5x5, 64, (5, 5), use_bias=False, name='mixed1_b22')

    branch3x3dbl = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed1_b31')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed1_b32')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed1_b33')
                     
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='mixed1_b41')(x)
    branch_pool = convolution_block(branch_pool, 64, (1, 1), use_bias=False, name='mixed1_b42')
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed2_b11')

    branch5x5 = convolution_block(x, 48, (1, 1), use_bias=False, name='mixed2_b21')
    branch5x5 = convolution_block(branch5x5, 64, (5, 5), use_bias=False, name='mixed2_b22')

    branch3x3dbl = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed2_b31')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed2_b32')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed2_b33')

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='mixed2_b41')(x)
    branch_pool = convolution_block(branch_pool, 64, (1, 1), use_bias=False, name='mixed2_b42')
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = convolution_block(x, 384, (3, 3), (2, 2), padding='valid', use_bias=False, name='mixed3_b11')

    branch3x3dbl = convolution_block(x, 64, (1, 1), use_bias=False, name='mixed3_b21')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), use_bias=False, name='mixed3_b22')
    branch3x3dbl = convolution_block(branch3x3dbl, 96, (3, 3), (2, 2), padding='valid', use_bias=False, name='mixed3_b23')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='mixed3_b31')(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = convolution_block(x, 192, (1, 1), use_bias=False, name='mixed4_b11')

    branch7x7 = convolution_block(x, 128, (1, 1), use_bias=False, name='mixed4_b21')
    branch7x7 = convolution_block(branch7x7, 128, (1, 7), use_bias=False, name='mixed4_b22')
    branch7x7 = convolution_block(branch7x7, 192, (7, 1), use_bias=False, name='mixed4_b23')

    branch7x7dbl = convolution_block(x, 128, (1, 1), use_bias=False, name='mixed4_b31')
    branch7x7dbl = convolution_block(branch7x7dbl, 128, (7, 1), use_bias=False, name='mixed4_b32')
    branch7x7dbl = convolution_block(branch7x7dbl, 128, (1, 7), use_bias=False, name='mixed4_b33')
    branch7x7dbl = convolution_block(branch7x7dbl, 128, (7, 1), use_bias=False, name='mixed4_b34')
    branch7x7dbl = convolution_block(branch7x7dbl, 192, (1, 7), use_bias=False, name='mixed4_b35')

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='mixed4_b41')(x)
    branch_pool = convolution_block(branch_pool, 192, (1, 1), use_bias=False, name='mixed4_b42')
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        name_stage = str(5 + i)
        branch1x1 = convolution_block(x, 192, (1, 1), use_bias=False, name=f'mixed{name_stage}_b11')

        branch7x7 = convolution_block(x, 160, (1, 1), use_bias=False, name=f'mixed{name_stage}_b21')
        branch7x7 = convolution_block(branch7x7, 160, (1, 7), use_bias=False, name=f'mixed{name_stage}_b22')
        branch7x7 = convolution_block(branch7x7, 192, (7, 1), use_bias=False, name=f'mixed{name_stage}_b23')

        branch7x7dbl = convolution_block(x, 160, (1, 1), use_bias=False, name=f'mixed{name_stage}_b31')
        branch7x7dbl = convolution_block(branch7x7dbl, 160, (7, 1), use_bias=False, name=f'mixed{name_stage}_b32')
        branch7x7dbl = convolution_block(branch7x7dbl, 160, (1, 7), use_bias=False, name=f'mixed{name_stage}_b33')
        branch7x7dbl = convolution_block(branch7x7dbl, 160, (7, 1), use_bias=False, name=f'mixed{name_stage}_b34')
        branch7x7dbl = convolution_block(branch7x7dbl, 192, (1, 7), use_bias=False, name=f'mixed{name_stage}_b35')

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=f'mixed{name_stage}_b41')(x)
        branch_pool = convolution_block(branch_pool, 192, (1, 1), use_bias=False, name=f'mixed{name_stage}_b42')
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name=f'mixed{name_stage}')

    # mixed 7: 17 x 17 x 768
    branch1x1 = convolution_block(x, 192, (1, 1), use_bias=False, name='mixed7_b11')

    branch7x7 = convolution_block(x, 192, (1, 1), use_bias=False, name='mixed7_b21')
    branch7x7 = convolution_block(branch7x7, 192, (1, 7), use_bias=False, name='mixed7_b22')
    branch7x7 = convolution_block(branch7x7, 192, (7, 1), use_bias=False, name='mixed7_b23')

    branch7x7dbl = convolution_block(x, 192, (1, 1), use_bias=False, name='mixed7_b31')
    branch7x7dbl = convolution_block(branch7x7dbl, 192, (7, 1), use_bias=False, name='mixed7_b32')
    branch7x7dbl = convolution_block(branch7x7dbl, 192, (1, 7), use_bias=False, name='mixed7_b33')
    branch7x7dbl = convolution_block(branch7x7dbl, 192, (7, 1), use_bias=False, name='mixed7_b34')
    branch7x7dbl = convolution_block(branch7x7dbl, 192, (1, 7), use_bias=False, name='mixed7_b35')

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='mixed7_b41')(x)
    branch_pool = convolution_block(branch_pool, 192, (1, 1), use_bias=False, name='mixed7_b42')
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = convolution_block(x, 192, (1, 1), use_bias=False, name='mixed8_b11')
    branch3x3 = convolution_block(branch3x3, 320, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='mixed8_b12')

    branch7x7x3 = convolution_block(x, 192, 1, 1, use_bias=False, name='mixed8_b21')
    branch7x7x3 = convolution_block(branch7x7x3, 192, (1, 7), use_bias=False, name='mixed8_b22')
    branch7x7x3 = convolution_block(branch7x7x3, 192, (7, 1), use_bias=False, name='mixed8_b23')
    branch7x7x3 = convolution_block(branch7x7x3, 192, (3, 3), use_bias=False, strides=(2, 2), padding='valid', name='mixed8_b24')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='mixed8_b31')(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        name_stage = str(9 + i)
        branch1x1 = convolution_block(x, 320, (1, 1), use_bias=False, name=f'mixed{name_stage}_b11')

        branch3x3 = convolution_block(x, 384, (1, 1), use_bias=False, name=f'mixed{name_stage}_b21')
        branch3x3_1 = convolution_block(branch3x3, 384, (1, 3), use_bias=False, name=f'mixed{name_stage}_b22')
        branch3x3_2 = convolution_block(branch3x3, 384, (3, 1), use_bias=False, name=f'mixed{name_stage}_b23')
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=-1, name=f'mixed{name_stage}_b24')

        branch3x3dbl = convolution_block(x, 448, (1, 1), use_bias=False, name=f'mixed{name_stage}_b31')
        branch3x3dbl = convolution_block(branch3x3dbl, 384, (3, 3), use_bias=False, name=f'mixed{name_stage}_b32')
        branch3x3dbl_1 = convolution_block(branch3x3dbl, 384, (1, 3), use_bias=False, name=f'mixed{name_stage}_b33')
        branch3x3dbl_2 = convolution_block(branch3x3dbl, 384, (3, 1), use_bias=False, name=f'mixed{name_stage}_b34')
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=-1, name=f'mixed{name_stage}_b35')

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=f'mixed{name_stage}_b41')(x)
        branch_pool = convolution_block(branch_pool, 192, (1, 1), use_bias=False, name=f'mixed{name_stage}_b42')
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name=f'mixed{name_stage}')
        
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name='Inception-v3')

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