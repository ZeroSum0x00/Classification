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
from models.layers import SplitWrapper, get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape



def Bottle2Neck(input_tensor, filters, stride=1, downsample=False, baseWidth=26, scale=4, activation="relu", normalizer='batch-norm', name=''):
    expansion = 4
    width = int(np.floor(filters * (baseWidth / 64.0)))
    nums = 1 if scale == 1 else scale - 1

    if downsample:
        residual = AveragePooling2D(pool_size=stride, strides=stride, padding='same', name=f'{name}.down.pool')(input_tensor)
        residual = Conv2D(filters=filters*expansion, kernel_size=1, strides=1, use_bias=False, name=f'{name}.down.conv')(residual)
        residual = get_normalizer_from_name(normalizer, name=f'{name}.down.norm')(residual)
        
    else:
        residual = input_tensor

    x = Conv2D(filters=width*scale, kernel_size=1, strides=1, use_bias=False, name=f'{name}.pre_conv')(input_tensor)
    x = get_normalizer_from_name(normalizer, name=f'{name}.pre_norm')(x)
    x = get_activation_from_name(activation, name=f'{name}.pre_activ')(x)
    
    spx = SplitWrapper(num_or_size_splits=scale, axis=-1)(x)
    for i in range(nums):
        
        if i == 0 or downsample:
            sp = spx[i]
        else:
            sp = sp + spx[i]
            
        sp = Conv2D(filters=width, kernel_size=3, strides=stride, padding="same", use_bias=False, name=f'{name}.layer{i}.conv')(sp)
        sp = get_normalizer_from_name(normalizer, name=f'{name}.layer{i}.norm')(sp)
        sp = get_activation_from_name(activation, name=f'{name}.layer{i}.activ')(sp)
        
        if i == 0:
            x = sp
        else:
            x = concatenate([x, sp], axis=-1)

    if scale != 1:
        if downsample:
            pool = AveragePooling2D(pool_size=(3, 3), strides=stride, padding='same', name=f'{name}.scale.pool')(spx[nums])
            x = concatenate([x, pool], axis=-1)
        else:
            x = concatenate([x, spx[nums]], axis=-1)

    x = Conv2D(filters=filters*expansion, kernel_size=1, strides=1, use_bias=False, name=f'{name}.post_conv')(x)
    x = get_normalizer_from_name(normalizer, name=f'{name}.post_norm')(x)
    x = add([x, residual])
    x = get_activation_from_name(activation, name=f'{name}.post_activ')(x)
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
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False, name='stem.block1.conv')(img_input)
    x = get_normalizer_from_name('batch-norm', name='stem.block1.norm')(x)
    x = get_activation_from_name('relu', name='stem.block1.activ')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='stem.block2.conv')(x)
    x = get_normalizer_from_name('batch-norm', name='stem.block2.norm')(x)
    x = get_activation_from_name('relu', name='stem.block2.activ')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name='stem.block3.conv')(x)
    x = get_normalizer_from_name('batch-norm', name='stem.block3.norm')(x)
    x = get_activation_from_name('relu', name='stem.block3.activ')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='stem.block3.pool')(x)

    # Block conv2_x
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        x = Bottle2Neck(x, 64, 1, downsaple, baseWidth, scale, name=f'stage1.block{i+1}')
    
    # Block conv3_x
    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        stride = 2 if i == 0 else 1
        x = Bottle2Neck(x, 128, stride, downsaple, baseWidth, scale, name=f'stage2.block{i+1}')

    # Block conv4_x
    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        stride = 2 if i == 0 else 1
        x = Bottle2Neck(x, 256, stride, downsaple, baseWidth, scale, name=f'stage3.block{i+1}')

    # Block conv5_x
    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = Bottle2Neck(x, 512, stride, downsaple, baseWidth, scale, name=f'stage4.block{i+1}')
    
    x = AveragePooling2D(pool_size=(24, 24), name='avg_pool')(x)

    # Final Block
    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(
            units=1 if num_classes == 2 else num_classes,
            activation=final_activation,
            name="predictions"
        )(x)
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
        y_2 = model.get_layer("stem.block3.activ").output
        y_4 = model.get_layer("stage2.block1.pre_activ").output
        y_8 = model.get_layer("stage3.block1.pre_activ").output
        y_16 = model.get_layer("stage4.block3.post_activ").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='Res2Net50_backbone')


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
        y_2 = model.get_layer("stem.block3.activ").output
        y_4 = model.get_layer("stage2.block1.pre_activ").output
        y_8 = model.get_layer("stage3.block1.pre_activ").output
        y_16 = model.get_layer("stage4.block3.post_activ").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='Res2Net101_backbone')


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
        y_2 = model.get_layer("stem.block3.activ").output
        y_4 = model.get_layer("stage2.block1.pre_activ").output
        y_8 = model.get_layer("stage3.block1.pre_activ").output
        y_16 = model.get_layer("stage4.block3.post_activ").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='Res2Net152_backbone')
