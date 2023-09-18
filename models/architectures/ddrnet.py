"""
  # Description:
    - The following table comparing the params of the DDRNet in Pytorch Source 
    with Tensorflow convert Source on size 224 x 224 x 3:
      
       ---------------------------------------------------------------------
      |    Library     |     Model Name      |    Params      |  Greater    |
      |-------------------------------------------------------|-------------|
      |   Pytorch      |    DDRNet23-slim    |   7,574,664    |     _       |
      |   Tensorflow   |    DDRNet23-slim    |   7,968,264    |    4.94(%)  |
      |----------------|---------------------|----------------|-------------|
      |   Pytorch      |       DDRNet23      |   28,216,616   |     _       |
      |   Tensorflow   |       DDRNet23      |   29,759,528   |    5.2(%)   |
      |----------------|---------------------|----------------|-------------|
      |   Pytorch      |       DDRNet39      |   40,130,344   |     _       |
      |   Tensorflow   |       DDRNet39      |   36,921,896   |    ?????    |
       ---------------------------------------------------------------------

  # Reference:
    - [Deep Dual-resolution Networks for Real-time and Accurate Semantic 
       Segmentation of Road Scenes](https://arxiv.org/pdf/2101.06085.pdf)
    - Source: https://github.com/ydhongHIT/DDRNet

"""

from __future__ import print_function

import warnings
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file
from utils.model_processing import _obtain_input_shape, correct_pad

try:
  from tensorflow.keras.layers.experimental import SyncBatchNormalization as BatchNorm
except ImportError:
  print('Can\'t import SyncBatchNormalization, using BatchNormalization')
  from tensorflow.keras.layers import BatchNormalization as BatchNorm
  
# TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
# TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def BasicBlock(input_tensor, filters, kernel_size=3, strides=1, downsaple=False):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    filter1, filter2 = filters
    shortcut = input_tensor

    x = Conv2D(filters=filter1, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(input_tensor)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter2, kernel_size=kernel_size, strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)

    if downsaple:
      shortcut = Conv2D(filters=filter2, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal', use_bias=False)(shortcut)
      shortcut = BatchNorm(axis=bn_axis, momentum=0.1)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def Bottleneck(input_tensor, filters, kernel_size, strides, downsaple=False):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    filter1, filter2, filter3 = filters
    shortcut = input_tensor

    x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(input_tensor)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter2, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)

    if downsaple:
      shortcut = Conv2D(filters=filter3, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal', use_bias=False)(shortcut)
      shortcut = BatchNorm(axis=bn_axis, momentum=0.1)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def bilateral_fusion(low_branch, high_branch, filters, up_size=2):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    filter1, filter2 = filters

    x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(low_branch)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=up_size, interpolation='bilinear')(x)
    x = add([high_branch, x])

    y = high_branch
    for i in range(up_size // 2):
        y = Conv2D(filters=filter2, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(y)
        y = BatchNorm(axis=bn_axis, momentum=0.1)(y)
        y = Activation('relu')(y)
    y = add([low_branch, y])
    return y, x


def DDRNet23_slim(include_top=True, 
                  weights='imagenet',
                  input_tensor=None, 
                  input_shape=None,
                  pooling=None,
                  final_activation="softmax",
                  classes=1000) -> Model:
                      
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
        bn_axis = -1
    else:
        bn_axis = 1

    # Branch conv1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(img_input)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    # Branch conv2
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = BasicBlock(x, [32, 32], 3, 1, True)
    x = BasicBlock(x, [32, 32], 3, 1, False)

    # Branch conv3
    x = BasicBlock(x, [64, 64], 3, 2, True)
    x = BasicBlock(x, [64, 64], 3, 1, False)

    # Branch conv4
    ## low branch
    low_branch = BasicBlock(x, [128, 128], 3, 2, True)
    low_branch = BasicBlock(low_branch, [128, 128], 3, 1, False)

    ## high branch
    high_branch = BasicBlock(x, [64, 64], 3, 1, True)
    high_branch = BasicBlock(high_branch, [64, 64], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [64, 128])

    # Branch conv5_1
    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 2, True)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = BasicBlock(high_branch, [64, 64], 3, 1, True)
    high_branch = BasicBlock(high_branch, [64, 64], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [64, 256], 4)

    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = Bottleneck(low_branch, [256, 256, 512], 3, 1, True)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = Bottleneck(high_branch, [64, 64, 128], 3, 1, True)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)

    # Branch conv5_2
    outputs = add([low_branch, high_branch])
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(outputs)
    outputs = BatchNorm(axis=bn_axis, momentum=0.1)(outputs)
    outputs = Activation('relu')(outputs)

    if include_top:
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(outputs)
    else:
        if pooling == 'avg':
            outputs = GlobalAveragePooling2D(name='avg_pool')(outputs)
        elif pooling == 'max':
            outputs = GlobalMaxPooling2D(name='max_pool')(outputs)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs=img_input, outputs=outputs, name='DDRNet-23-slim')
                      
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = None
        else:
            weights_path = None
            
        if weights_path is not None:
            model.load_weights(weights_path)
            
    elif weights is not None:
        model.load_weights(weights)
        
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


def DDRNet23(include_top=True, 
             weights='imagenet',
             input_tensor=None, 
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000) -> Model:
                 
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
        bn_axis = -1
    else:
        bn_axis = 1

    # Branch conv1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(img_input)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    # Branch conv2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = BasicBlock(x, [64, 64], 3, 1, True)
    x = BasicBlock(x, [64, 64], 3, 1, False)

    # Branch conv3
    x = BasicBlock(x, [128, 128], 3, 2, True)
    x = BasicBlock(x, [128, 128], 3, 1, False)

    # Branch conv4
    ## low branch
    low_branch = BasicBlock(x, [256, 256], 3, 2, True)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)

    ## high branch
    high_branch = BasicBlock(x, [128, 128], 3, 1, True)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [128, 256])

    # Branch conv5_1
    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = BasicBlock(low_branch, [512, 512], 3, 2, True)
    low_branch = BasicBlock(low_branch, [512, 512], 3, 1, False)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, True)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [128, 512], 4)

    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = Bottleneck(low_branch, [512, 512, 1024], 3, 1, True)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = Bottleneck(high_branch, [128, 128, 256], 3, 1, True)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)

    # Branch conv5_2
    outputs = add([low_branch, high_branch])
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(outputs)
    outputs = BatchNorm(axis=bn_axis, momentum=0.1)(outputs)
    outputs = Activation('relu')(outputs)

    if include_top:
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(outputs)
    else:
        if pooling == 'avg':
            outputs = GlobalAveragePooling2D(name='avg_pool')(outputs)
        elif pooling == 'max':
            outputs = GlobalMaxPooling2D(name='max_pool')(outputs)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs=img_input, outputs=outputs, name='DDRNet-23')
                 
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = None
        else:
            weights_path = None
            
        if weights_path is not None:
            model.load_weights(weights_path)
            
    elif weights is not None:
        model.load_weights(weights)
        
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


def DDRNet39(include_top=True, 
             weights='imagenet',
             input_tensor=None, 
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000) -> Model:
                 
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
        bn_axis = -1
    else:
        bn_axis = 1

    # Branch conv1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(img_input)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    # Branch conv2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNorm(axis=bn_axis, momentum=0.1)(x)
    x = Activation('relu')(x)

    x = BasicBlock(x, [64, 64], 3, 1, True)
    x = BasicBlock(x, [64, 64], 3, 1, False)
    x = BasicBlock(x, [64, 64], 3, 1, False)

    # Branch conv3
    x = BasicBlock(x, [128, 128], 3, 2, True)
    x = BasicBlock(x, [128, 128], 3, 1, False)
    x = BasicBlock(x, [128, 128], 3, 1, False)
    x = BasicBlock(x, [128, 128], 3, 1, False)

    # Branch conv4
    ## low branch
    low_branch = BasicBlock(x, [256, 256], 3, 2, True)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)

    ## high branch
    high_branch = BasicBlock(x, [128, 128], 3, 1, True)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [128, 256])

    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = BasicBlock(x, [256, 256], 3, 2, True)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)
    low_branch = BasicBlock(low_branch, [256, 256], 3, 1, False)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = BasicBlock(x, [128, 128], 3, 1, True)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [128, 256])

    # Branch conv5_1
    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = BasicBlock(low_branch, [512, 512], 3, 2, True)
    low_branch = BasicBlock(low_branch, [512, 512], 3, 1, False)
    low_branch = BasicBlock(low_branch, [512, 512], 3, 1, False)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, True)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)
    high_branch = BasicBlock(high_branch, [128, 128], 3, 1, False)

    ## bilateral fusion branch
    low_branch, high_branch = bilateral_fusion(low_branch, high_branch, [128, 512], 4)

    ## low branch
    low_branch = Activation('relu')(low_branch)
    low_branch = Bottleneck(low_branch, [512, 512, 1024], 3, 1, True)

    ## high branch
    high_branch = Activation('relu')(high_branch)
    high_branch = Bottleneck(high_branch, [128, 128, 256], 3, 1, True)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)
    high_branch = Activation('relu')(high_branch)
    high_branch = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(high_branch)
    high_branch = BatchNorm(axis=bn_axis, momentum=0.1)(high_branch)

    # Branch conv5_2
    outputs = add([low_branch, high_branch])
    outputs = Activation('relu')(outputs)
    outputs = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', use_bias=False)(outputs)
    outputs = BatchNorm(axis=bn_axis, momentum=0.1)(outputs)
    outputs = Activation('relu')(outputs)

    if include_top:
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(outputs)
    else:
        if pooling == 'avg':
            outputs = GlobalAveragePooling2D(name='avg_pool')(outputs)
        elif pooling == 'max':
            outputs = GlobalMaxPooling2D(name='max_pool')(outputs)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs=img_input, outputs=outputs, name='DDRNet-39')
                 
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = None
        else:
            weights_path = None
            
        if weights_path is not None:
            model.load_weights(weights_path)
            
    elif weights is not None:
        model.load_weights(weights)
        
    if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image data format convention '
                      '(`image_data_format="channels_first"`). '
                      'For best performance, set '
                      '`image_data_format="channels_last"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')


if __name__ == '__main__':
    model = DDRNet23_slim(include_top=False, weights=None)
    model.summary()