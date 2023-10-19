"""
  # Description:
    - The following table comparing the params of the SqueezeNet in Tensorflow on 
    size 224 x 224 x 3:

       -------------------------------------
      |     Model Name    |    Params       |
      |-------------------------------------|
      |    SqueezeNet     |    2,237,904    |
       -------------------------------------

  # Reference:
    - [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/pdf/1602.07360.pdf)
    - Source: https://github.com/guaiguaibao/GoogLeNet_Tensorflow2.0/tree/master/tensorflow2.0/GoogLeNet
              https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/SqueezeNet.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name
from utils.model_processing import _obtain_input_shape


def fire_module(inputs, filters, squeeze_channel, activation='relu'):
    x = Conv2D(filters=squeeze_channel, kernel_size=(1, 1), strides=(1, 1), padding="valid")(inputs)
    x = get_activation_from_name(activation)(x)
    expand_1x1 = Sequential([
        Conv2D(filters=filters // 2, kernel_size=(1, 1), strides=(1, 1), padding="valid"),
        get_activation_from_name(activation)
    ])(x)
    expand_3x3 = Sequential([
        Conv2D(filters=filters // 2, kernel_size=(3, 3), strides=(1, 1), padding="same"),
        get_activation_from_name(activation)
    ])(x)
    return concatenate([expand_1x1, expand_3x3])


def SqueezeNet(include_top=True, 
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1    

    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(img_input)
    x = get_activation_from_name('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 128, 16)
    x = fire_module(x, 128, 16)
    x = fire_module(x, 256, 32)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 256, 32)
    x = fire_module(x, 384, 48)
    x = fire_module(x, 384, 48)
    x = fire_module(x, 512, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 512, 64)
    x = Conv2D(filters=1 if classes == 2 else classes, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = AveragePooling2D((13, 13), strides=(1, 1))(x)
                   
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='SqueezeNet')

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