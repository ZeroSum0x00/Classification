"""
  # Description:
    - The following table comparing the params of the Inception v1 (GoogleNet) in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------------
      |        Model Name         |    Params       |
      |---------------------------------------------|
      |    Inception v1 naive     |   436,894,728   |
      |---------------------------------------------|
      |       Inception v1        |    56,146,392   |
       ---------------------------------------------

  # Reference:
    - [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
    - Source: https://github.com/guaiguaibao/GoogLeNet_Tensorflow2.0/tree/master/tensorflow2.0/GoogLeNet
              https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/GoogLeNet.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def inception_v1_naive_block(inputs, blocks, activation='relu'):
    """
    Inception module, naive version

    :param inputs: input tensor.
    :param blocks:
    :return: Output tensor for the block.
    """
    branch1 = Conv2D(filters=blocks[0], kernel_size=(1, 1), strides=(1, 1))(inputs)
    branch1 = get_activation_from_name(activation)(branch1)

    branch2 = Conv2D(filters=blocks[1], kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch2 = get_activation_from_name(activation)(branch2)

    branch3 = Conv2D(filters=blocks[2], kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    branch3 = get_activation_from_name(activation)(branch3)
    
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(inputs)

    merger = concatenate([branch1, branch2, branch3, branch4])
    return merger


def inception_v1_full_block(inputs, blocks, activation='relu'):
    """
    Inception module with dimension reductions

    :param inputs: input tensor.
    :param blocks: filter block, respectively: #1×1, #3×3 reduce, #3×3, #5×5 reduce, #5×5, pool proj
    :return: Output tensor for the block.
    """
    branch_1x1 = Conv2D(filters=blocks[0], kernel_size=(1, 1))(inputs)
    branch_1x1 = get_activation_from_name(activation)(branch_1x1)

    branch_3x3 = Conv2D(filters=blocks[1], kernel_size=(1, 1))(inputs)
    branch_3x3 = get_activation_from_name(activation)(branch_3x3)
    branch_3x3 = Conv2D(filters=blocks[2], kernel_size=(3, 3), padding='same')(branch_3x3)
    branch_3x3 = get_activation_from_name(activation)(branch_3x3)

    branch_5x5 = Conv2D(filters=blocks[3], kernel_size=(1, 1))(inputs)
    branch_5x5 = get_activation_from_name(activation)(branch_5x5)
    branch_5x5 = Conv2D(filters=blocks[4], kernel_size=(5, 5), padding='same')(branch_5x5)
    branch_5x5 = get_activation_from_name(activation)(branch_5x5)

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv2D(filters=blocks[5], kernel_size=(1, 1))(branch_pool)
    branch_pool = get_activation_from_name(activation)(branch_pool)

    merger = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool])
    return merger


def inception_v1_auxiliary_block(inputs,  num_classes, activation='relu', final_activation='softmax'):
    """
    Inception auxiliary classifier module

    :param inputs: input tensor.
    :param num_classes: number off classes
    :return: Output tensor for the block.
    """
    x = AveragePooling2D(pool_size=5, strides=3)(inputs)
    x = Conv2D(filters=128, kernel_size=(1, 1), padding='valid')(x)
    x = get_activation_from_name(activation)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=1024)(x)
    x = get_activation_from_name(activation)(x)
    x = Dense(units=num_classes)(x)
    x = get_activation_from_name(final_activation)(x)
    return x


def GoogleNet(block,
              auxiliary_logits=False,
              include_top=True, 
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

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(img_input)
    x = get_activation_from_name('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(filters=192, kernel_size=(3, 3), padding='same')(x)
    x = get_activation_from_name('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = block(x, [64, 96, 128, 16, 32, 32])
    x = block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = block(x, [192, 96, 208, 16, 48, 64])
                  
    if auxiliary_logits:
        aux1 = inception_v1_auxiliary_block(x, classes)
        
    x = block(x, [160, 112, 224, 24, 64, 64])
    x = block(x, [128, 128, 256, 24, 64, 64])
    x = block(x, [112, 144, 288, 32, 64, 64])
                  
    if auxiliary_logits:
        aux2 = inception_v1_auxiliary_block(x, classes)
        
    x = block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = block(x, [256, 160, 320, 32, 128, 128])
    x = block(x, [384, 192, 384, 48, 128, 128])

    if include_top:
        # Classification block
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(x)
        x = Dropout(rate=0.4)(x)
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
    if auxiliary_logits:
        model = Model(inputs, [aux1, aux2, x], name='Inception-v1')
    else:
        model = Model(inputs, x, name='Inception-v1')

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


def Inception_v1_naive(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       final_activation="softmax",
                       classes=1000) -> Model:
    
    model = GoogleNet(block=inception_v1_naive_block,
                      auxiliary_logits=False,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      final_activation=final_activation,
                      classes=classes)
    return model


def Inception_v1(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000) -> Model:
    
    model = GoogleNet(block=inception_v1_full_block,
                      auxiliary_logits=False,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      final_activation=final_activation,
                      classes=classes)
    return model