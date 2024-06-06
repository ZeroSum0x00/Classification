"""
  # Description:
    - The following table comparing the params of the Res2Net in Tensorflow on 
    size 384 x 384 x 3:

       -------------------------------------------------------
      |     Model Name      |    Setting     |    Params      |
      |---------------------|----------------|----------------|
      |     Res2Net-50      |    26w x 4s    |   25,777,972   |
      |---------------------|----------------|----------------|
      |     Res2Net-101     |    26w x 4s    |   45,345,108   |
      |---------------------|----------------|----------------|
      |     Res2Net-152     |    26w x 4s    |   25,777,972   |
       -------------------------------------------------------

  # Reference:
    - [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf)
    - Source: https://github.com/Res2Net/Res2Net-PretrainedModels

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape



def Bottle2Neck(input_tensor, filters, stride=1, downsample=False, baseWidth=26, scale=4, activation="relu", normalizer='batch-norm'):
    expansion = 4
    width = int(np.floor(filters * (baseWidth / 64.0)))
    nums = 1 if scale == 1 else scale - 1

    if downsample:
        residual = AveragePooling2D(pool_size=stride, strides=stride, padding='same')(input_tensor)
        residual = Conv2D(filters=filters*expansion, kernel_size=1, strides=1, use_bias=False)(residual)
        residual = get_normalizer_from_name(normalizer)(residual)
        
    else:
        residual = input_tensor

    x = Conv2D(filters=width*scale, kernel_size=1, strides=1, use_bias=False)(input_tensor)
    x = get_normalizer_from_name(normalizer)(x)
    x = get_activation_from_name(activation)(x)
    
    spx = tf.split(x, scale, axis=-1)
    for i in range(nums):
        
        if i == 0 or downsample:
            sp = spx[i]
        else:
            sp = sp + spx[i]
            
        sp = Conv2D(filters=width, kernel_size=3, strides=stride, padding="same", use_bias=False)(sp)
        sp = get_normalizer_from_name(normalizer)(sp)
        sp = get_activation_from_name(activation)(sp)
        
        if i == 0:
            x = sp
        else:
            x = concatenate([x, sp], axis=-1)

    if scale != 1:
        if downsample:
            pool = AveragePooling2D(pool_size=(3, 3), strides=stride, padding='same')(spx[nums])
            x = concatenate([x, pool], axis=-1)
        else:
            x = concatenate([x, spx[nums]], axis=-1)

    x = Conv2D(filters=filters*expansion, kernel_size=1, strides=1, use_bias=False)(x)
    x = get_normalizer_from_name(normalizer)(x)
    x = add([x, residual])
    x = get_activation_from_name(activation)(x)
    return x


def Res2Net(num_blocks,
            baseWidth=26,
            scale=4,
            include_top=True, 
            weights='imagenet',
            input_tensor=None, 
            input_shape=None,
            pooling=None,
            final_activation="softmax",
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
                                      default_size=384,
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

    # Block conv1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = get_normalizer_from_name('batch-norm')(x)
    x = get_activation_from_name('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = get_normalizer_from_name('batch-norm')(x)
    x = get_activation_from_name('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = get_normalizer_from_name('batch-norm')(x)
    x = get_activation_from_name('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Block conv2_x
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        x = Bottle2Neck(x, 64, 1, downsaple, baseWidth, scale)
    
    # Block conv3_x
    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        stride = 2 if i == 0 else 1
        x = Bottle2Neck(x, 128, stride, downsaple, baseWidth, scale)

    # Block conv4_x
    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        stride = 2 if i == 0 else 1
        x = Bottle2Neck(x, 256, stride, downsaple, baseWidth, scale)

    # Block conv5_x
    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = Bottle2Neck(x, 512, stride, downsaple, baseWidth, scale)
    
    x = AveragePooling2D(pool_size=(24, 24), name='avg_pool')(x)

    # Final Block
    if include_top:
        x = Flatten(name='flatten')(x)
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

    # Create model.
    if num_blocks == [3, 4, 6, 3]:
        model = Model(inputs, x, name='Res2Net-50')
    elif num_blocks == [3, 4, 23, 3]:
        model = Model(inputs, x, name='Res2Net-101')
    elif num_blocks == [3, 8, 36, 3]:
        model = Model(inputs, x, name='Res2Net-152')
    else:
        model = Model(inputs, x, name='Res2Net')

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


def Res2Net50(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000) -> Model:
    
    model = Res2Net(num_blocks=[3, 4, 6, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def Res2Net50_backbone(input_shape=(384, 384, 3),
                       include_top=False, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       final_activation="softmax",
                       classes=1000,
                       custom_layers=None) -> Model:

    model = Res2Net50(include_top=include_top,
                     weights=weights,
                     input_tensor=input_tensor, 
                     input_shape=input_shape,
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='Res2Net50_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c6_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='Res2Net50_backbone')


def Res2Net101(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000) -> Model:
    
    model = Res2Net(num_blocks=[3, 4, 23, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def Res2Net101_backbone(input_shape=(384, 384, 3),
                        include_top=False, 
                        weights='imagenet', 
                        input_tensor=None, 
                        pooling=None, 
                        final_activation="softmax",
                        classes=1000,
                        custom_layers=None) -> Model:

    model = Res2Net101(include_top=include_top,
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      final_activation=final_activation,
                      classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='Res2Net101_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c6_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='Res2Net101_backbone')


def Res2Net152(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000) -> Model:
    
    model = Res2Net(num_blocks=[3, 8, 36, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def Res2Net152_backbone(input_shape=(384, 384, 3),
                        include_top=False, 
                        weights='imagenet', 
                        input_tensor=None, 
                        pooling=None, 
                        final_activation="softmax",
                        classes=1000,
                        custom_layers=None) -> Model:

    model = Res2Net152(include_top=include_top,
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      final_activation=final_activation,
                      classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='Res2Net152_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c6_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='Res2Net152_backbone')
