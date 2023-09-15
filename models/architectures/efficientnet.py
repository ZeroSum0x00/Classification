"""
  # Description:
    - The following table comparing the params of the EfficientNet in Tensorflow on 
    size 224 x 224 x 3:

       -----------------------------------------
      |     Model Name         |    Params      |
      |-----------------------------------------|
      |     EfficientNet-B0    |   8,062,504    |
      |------------------------|----------------|
      |     EfficientNet-B1    |   14,307,880   |
      |------------------------|----------------|
      |     EfficientNet-B2    |   20,242,984   |
      |------------------------|----------------|
      |     EfficientNet-B3    |   33,736,232   |
      |------------------------|----------------|
      |     EfficientNet-B4    |   33,736,232   |
      |------------------------|----------------|
      |     EfficientNet-B5    |   33,736,232   |
      |------------------------|----------------|
      |     EfficientNet-B6    |   33,736,232   |
      |------------------------|----------------|
      |     EfficientNet-B7    |   33,736,232   |
       -----------------------------------------

  # Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946v5.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
              https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

"""

from __future__ import print_function
from __future__ import absolute_import

import math
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file
from utils.model_processing import _obtain_input_shape, correct_pad


BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}


DEFAULT_BLOCKS_ARGS = [
    {'filters_in': 32, 'filters_out': 16, 'kernel_size': 3, 'strides': 1, 
     'expand_ratio': 1, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 1},
    {'filters_in': 16, 'filters_out': 24, 'kernel_size': 3, 'strides': 2, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 2},
    {'filters_in': 24, 'filters_out': 40, 'kernel_size': 5, 'strides': 2, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 2},
    {'filters_in': 40, 'filters_out': 80, 'kernel_size': 3, 'strides': 2, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 3},
    {'filters_in': 80, 'filters_out': 112, 'kernel_size': 5, 'strides': 1, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 3},
    {'filters_in': 112, 'filters_out': 192, 'kernel_size': 5, 'strides': 2, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 4},
    {'filters_in': 192, 'filters_out': 320, 'kernel_size': 3, 'strides': 1, 
     'expand_ratio': 6, 'squeeze_ratio': 0.25, 'residual_connection': True, 'repeats': 1},
]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def config_model_extractor(args, width_coefficient, depth_divisor=8):
    filters_in = round_filters(args['filters_in'], width_coefficient, depth_divisor)
    filters_out = round_filters(args['filters_out'], width_coefficient, depth_divisor)
    kernel_size = args['kernel_size']
    strides = args['strides']
    expand_ratio = args['expand_ratio']
    squeeze_ratio = args['squeeze_ratio']
    residual_connection = args['residual_connection']
    repeats = args['repeats']
    return filters_in, filters_out, kernel_size, strides, expand_ratio, squeeze_ratio, residual_connection, repeats


def swish(x):
    if K.backend() == 'tensorflow':
        try:
            return tf.nn.swish(x)
        except AttributeError:
            pass
    return x * tf.nn.sigmoid(x)

def EfficientBlock(inputs, 
                   filters_in,
                   filters_out,
                   kernel_size, 
                   strides,
                   expand_ratio=1, 
                   squeeze_ratio=0., 
                   activation=swish, 
                   residual_connection=True, 
                   drop_rate=0., 
                   name='efficien_block'):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    filters = filters_in * expand_ratio
    
    if expand_ratio != 1:
        x = Conv2D(filters=filters,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same',
                   activation=activation,
                   use_bias=False,
                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                   name=name + 'expand_conv')(inputs)
        x = BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(x, kernel_size), name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    x = DepthwiseConv2D(kernel_size=kernel_size,
                        strides=strides,
                        padding=conv_pad,
                        use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'dwconv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = Activation(activation, name=name + 'activation')(x)

    if 0 < squeeze_ratio <= 1:
        filters_squeeze = max(1, int(filters_in * squeeze_ratio))
        squeeze = GlobalAveragePooling2D(name=name + 'se_squeeze')(x)

        if bn_axis == 1:
            squeeze = Reshape((filters, 1, 1), name=name + 'se_reshape')(squeeze)
        else:
            squeeze = Reshape((1, 1, filters), name=name + 'se_reshape')(squeeze)

        squeeze = Conv2D(filters=filters_squeeze,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation=activation,
                         kernel_initializer=CONV_KERNEL_INITIALIZER,
                         name=name + 'se_reduce')(squeeze)
        squeeze = Conv2D(filters=filters,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='sigmoid',
                         kernel_initializer=CONV_KERNEL_INITIALIZER,
                         name=name + 'se_expand')(squeeze)
        x = multiply([x, squeeze], name=name + 'se_excite')
    
    x = Conv2D(filters=filters_out,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'project_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)

    if (residual_connection is True) and (strides == 1) and (filters_in == filters_out):
        if drop_rate > 0:
            x = Dropout(rate=drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = add([x, inputs], name=name + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 activation=swish,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 drop_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=input_shape[0],
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

    x = ZeroPadding2D(padding=correct_pad(img_input, 3), name='stem_conv_pad')(img_input)
    x = Conv2D(filters=round_filters(32, width_coefficient, depth_divisor),
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='valid',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name='stem_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = Activation(activation, name='stem_activation')(x)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for idx, args in enumerate(blocks_args):
        filters_in, filters_out, kernel_size, strides, expand_ratio, squeeze_ratio, residual_connection, repeats = config_model_extractor(args, width_coefficient, depth_divisor)
        for i in range(round_repeats(repeats, depth_coefficient)):
            if i > 0:
                strides = 1
                filters_in = filters_out

            x = EfficientBlock(x, 
                               filters_in, 
                               filters_out, 
                               kernel_size, 
                               strides, 
                               expand_ratio, 
                               squeeze_ratio, 
                               activation, 
                               residual_connection,
                               drop_connect_rate * b / blocks,
                               name='block_{}{}_'.format(idx + 1, chr(i + 97)))
            b += 1

    x = Conv2D(round_filters(1280, width_coefficient, depth_divisor),
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name='top_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = Activation(activation, name='top_activation')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        if drop_rate > 0:
            x = Dropout(drop_rate, name='top_dropout')(x)
        x = Dense(1 if classes == 2 else classes, 
                  activation=final_activation, 
                  kernel_initializer=DENSE_KERNEL_INITIALIZER, 
                  name='predictions')(x)
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

    # Create model.
    if (width_coefficient == 1.0) and (depth_coefficient == 1.0):
        model = Model(inputs, x, name='EfficientNet-B0')
    elif (width_coefficient == 1.0) and (depth_coefficient == 1.1):
        model = Model(inputs, x, name='EfficientNet-B1')
    elif (width_coefficient == 1.1) and (depth_coefficient == 1.2):
        model = Model(inputs, x, name='EfficientNet-B2')
    elif (width_coefficient == 1.2) and (depth_coefficient == 1.4):
        model = Model(inputs, x, name='EfficientNet-B3')
    elif (width_coefficient == 1.4) and (depth_coefficient == 1.8):
        model = Model(inputs, x, name='EfficientNet-B4')
    elif (width_coefficient == 1.6) and (depth_coefficient == 2.2):
        model = Model(inputs, x, name='EfficientNet-B5')
    elif (width_coefficient == 1.8) and (depth_coefficient == 2.6):
        model = Model(inputs, x, name='EfficientNet-B6')
    elif (width_coefficient == 2.0) and (depth_coefficient == 3.1):
        model = Model(inputs, x, name='EfficientNet-B7')
    else:
        model = Model(inputs, x, name='EfficientNet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if (width_coefficient == 1.0) and (depth_coefficient == 1.0):
                file_name = 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='e9e877068bd0af75e0a36691e03c072c')
            elif (width_coefficient == 1.0) and (depth_coefficient == 1.1):
                file_name = 'efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='8f83b9aecab222a9a2480219843049a1')
            elif (width_coefficient == 1.1) and (depth_coefficient == 1.2):
                file_name = 'efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='b6185fdcd190285d516936c09dceeaa4')
            elif (width_coefficient == 1.2) and (depth_coefficient == 1.4):
                file_name = 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='b2db0f8aac7c553657abb2cb46dcbfbb')
            elif (width_coefficient == 1.4) and (depth_coefficient == 1.8):
                file_name = 'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='ab314d28135fe552e2f9312b31da6926')
            elif (width_coefficient == 1.6) and (depth_coefficient == 2.2):
                file_name = 'efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='8d60b903aff50b09c6acf8eaba098e09')
            elif (width_coefficient == 1.8) and (depth_coefficient == 2.6):
                file_name = 'efficientnet-b6_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='a967457886eac4f5ab44139bdd827920')
            elif (width_coefficient == 2.0) and (depth_coefficient == 3.1):
                file_name = 'efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='e964fd6e26e9a4c144bcb811f2a10f20')
        else:
            if (width_coefficient == 1.0) and (depth_coefficient == 1.0):
                file_name = 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='345255ed8048c2f22c793070a9c1a130')
            elif (width_coefficient == 1.0) and (depth_coefficient == 1.1):
                file_name = 'efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='b20160ab7b79b7a92897fcb33d52cc61')
            elif (width_coefficient == 1.1) and (depth_coefficient == 1.2):
                file_name = 'efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='c6e46333e8cddfa702f4d8b8b6340d70')
            elif (width_coefficient == 1.2) and (depth_coefficient == 1.4):
                file_name = 'efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='e0cf8654fad9d3625190e30d70d0c17d')
            elif (width_coefficient == 1.4) and (depth_coefficient == 1.8):
                file_name = 'efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='b46702e4754d2022d62897e0618edc7b')
            elif (width_coefficient == 1.6) and (depth_coefficient == 2.2):
                file_name = 'efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='0a839ac36e46552a881f2975aaab442f')
            elif (width_coefficient == 1.8) and (depth_coefficient == 2.6):
                file_name = 'efficientnet-b6_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='375a35c17ef70d46f9c664b03b4437f2')
            elif (width_coefficient == 2.0) and (depth_coefficient == 3.1):
                file_name = 'efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
                weights_path = get_file(
                    file_name,
                    BASE_WEIGHTS_PATH + file_name,
                    cache_subdir='models',
                    file_hash='d55674cc46b805f4382d18bc08ed43c1')
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


def EfficientNetB0(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.2,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.0,
                         depth_coefficient=1.0,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.2,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.2,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.0,
                         depth_coefficient=1.1,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.2,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.3,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.1,
                         depth_coefficient=1.2,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.3,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.3,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.2,
                         depth_coefficient=1.4,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.3,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.4,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.4,
                         depth_coefficient=1.8,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.4,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.4,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.6,
                         depth_coefficient=2.2,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.4,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.5,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=1.8,
                         depth_coefficient=2.6,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.5,
                         drop_connect_rate=0.2)
    return model


def EfficientNetB7(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   final_activation="softmax",
                   classes=1000,
                   depth_divisor=8,
                   drop_rate=0.5,
                   drop_connect_rate=0.2) -> Model:
    
    model = EfficientNet(width_coefficient=2.0,
                         depth_coefficient=3.1,
                         blocks_args=DEFAULT_BLOCKS_ARGS,
                         activation=swish,
                         include_top=include_top,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=pooling,
                         final_activation=final_activation,
                         classes=classes,
                         depth_divisor=8,
                         drop_rate=0.5,
                         drop_connect_rate=0.2)
    return model
