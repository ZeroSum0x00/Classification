"""
  # Description:
    - The following table comparing the params of the ResNet in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------
      |     Model Name      |    Params      |
      |--------------------------------------|
      |     ResNet-18       |   11,708,328   |
      |---------------------|----------------|
      |     ResNet-34       |   21,827,624   |
      |---------------------|----------------|
      |     ResNet-50       |   25,636,712   |
      |---------------------|----------------|
      |     ResNet-101      |   44,707,176   |
      |---------------------|----------------|
      |     ResNet-152      |   60,419,944   |
       --------------------------------------

  # Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name, get_nomalizer_from_name
from utils.model_processing import _obtain_input_shape


RESNET50_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def BasicBlock(input_tensor, filters, kernel_size=3, downsaple=False, activation="relu", normalizer='batch-norm', stage='a', block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    prefix = 'residual_block_' + stage + str(block)
    filter1, filter2 = filters
    if downsaple and stage != 'a':
        strides = 2
    else:
        strides = 1
        
    shortcut = input_tensor

    x = Conv2D(filters=filter1, kernel_size=kernel_size, strides=strides, padding='same', name=prefix + '_conv1')(input_tensor)
    x = get_nomalizer_from_name(normalizer, name=prefix + '_batchnorm1')(x)
    x = get_activation_from_name(activation, name=prefix + '_activation1')(x)
    
    x = Conv2D(filters=filter2, kernel_size=kernel_size, strides=(1, 1), padding='same', name=prefix + '_conv2')(x)
    x = get_nomalizer_from_name(normalizer, name=prefix + '_batchnorm2')(x)

    if downsaple:
      shortcut = Conv2D(filters=filter2, kernel_size=(1, 1), strides=strides, name=prefix + '_shortcut')(shortcut)
      shortcut = get_nomalizer_from_name(normalizer, name=prefix + '_shortcut_batchnorm')(shortcut)

    x = add([x, shortcut], name=prefix + '_merge')
    x = get_activation_from_name(activation, name=prefix + '_final')(x)
    return x


def Bottleneck(input_tensor, filters, kernel_size=3, downsaple=False, activation="relu", normalizer='batch-norm', stage='a', block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    prefix = 'residual_block_' + stage + str(block)
    filter1, filter2, filter3 = filters
    if downsaple and stage != 'a':
        strides = 2
    else:
        strides = 1   
        
    shortcut = input_tensor

    x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), name=prefix + '_conv1')(input_tensor)
    x = get_nomalizer_from_name(normalizer, name=prefix + '_batchnorm1')(x)
    x = get_activation_from_name(activation, name=prefix + '_activation1')(x)
    
    x = Conv2D(filters=filter2, kernel_size=kernel_size, strides=strides, padding='same', name=prefix + '_conv2')(x)
    x = get_nomalizer_from_name(normalizer, name=prefix + '_batchnorm2')(x)
    x = get_activation_from_name(activation, name=prefix + '_activation2')(x)
    
    x = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(1, 1), name=prefix + '_conv3')(x)
    x = get_nomalizer_from_name(normalizer, name=prefix + '_batchnorm3')(x)

    if downsaple:
      shortcut = Conv2D(filters=filter3, kernel_size=(1, 1), strides=strides, name=prefix + '_shortcut')(shortcut)
      shortcut = get_nomalizer_from_name(normalizer, name=prefix + '_shortcut_batchnorm')(shortcut)

    x = add([x, shortcut], name=prefix + '_merge')
    x = get_activation_from_name(activation, name=prefix + '_final')(x)
    return x


def ResNetA(num_blocks,
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
    
    # Block conv1
    x = ZeroPadding2D(padding=(3, 3), name='initial_block_zeropadding')(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='initial_block_conv')(x)
    x = get_nomalizer_from_name('batch-norm', name='initial_block_batchnorm')(x)
    x = get_activation_from_name('relu', name='initial_block_activation')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_maxpool')(x)

    # Block conv2_x
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [64, 64], 3, downsaple, stage='a', block=i+1)
    
    # Block conv3_x
    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [128, 128], 3, downsaple, stage='b', block=i+1)

    # Block conv4_x
    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [256, 256], 3, downsaple, stage='c', block=i+1)

    # Block conv5_x
    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = BasicBlock(x, [512, 512], 3, downsaple, stage='d', block=i+1)
    
    outputs = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(x)

    # Final Block
    if include_top:
        outputs = Flatten(name='flatten')(outputs)
        outputs = Dense(1 if classes == 2 else classes, name='predictions')(outputs)
        outputs = get_activation_from_name(final_activation)(outputs)
    else:
        if pooling == 'avg':
            outputs = GlobalAveragePooling2D(name='global_avgpool')(outputs)
        elif pooling == 'max':
            outputs = GlobalMaxPooling2D(name='global_maxpool')(outputs)
            
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    if num_blocks == [2, 2, 2, 2]:
        model = Model(inputs, outputs, name='ResNet-18')
    elif num_blocks == [3, 4, 6, 3]:
        model = Model(inputs, outputs, name='ResNet-34')
    else:
        model = Model(inputs, outputs, name='ResNet-A')
    
    # Load weights.
    if weights == 'imagenet':
        
        if include_top:
            if num_blocks == [2, 2, 2, 2]:
                weights_path = None
            elif num_blocks == [3, 4, 6, 3]:
                weights_path = None
        else:
            if num_blocks == [2, 2, 2, 2]:
                weights_path = None
            elif num_blocks == [3, 4, 6, 3]:
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


def ResNetB(num_blocks,
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
    
    # Block conv1
    x = ZeroPadding2D(padding=(3, 3), name='initial_block_zeropadding')(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='initial_block_conv')(x)
    x = get_nomalizer_from_name('batch-norm', name='initial_block_batchnorm')(x)
    x = get_activation_from_name('relu', name='initial_block_activation')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_maxpool')(x)

    # Block conv2_x
    for i in range(num_blocks[0]):
        downsaple = True if i == 0 else False
        x = Bottleneck(x, [64, 64, 256], 3, downsaple, stage='a', block=i+1)
    
    # Block conv3_x
    for i in range(num_blocks[1]):
        downsaple = True if i == 0 else False
        x = Bottleneck(x, [128, 128, 512], 3, downsaple, stage='b', block=i+1)

    # Block conv4_x
    for i in range(num_blocks[2]):
        downsaple = True if i == 0 else False
        x = Bottleneck(x, [256, 256, 1024], 3, downsaple, stage='c', block=i+1)

    # Block conv5_x
    for i in range(num_blocks[3]):
        downsaple = True if i == 0 else False
        x = Bottleneck(x, [512, 512, 2048], 3, downsaple, stage='d', block=i+1)
    
    outputs = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(x)

    # Final Block
    if include_top:
        outputs = Flatten(name='flatten')(outputs)
        outputs = Dense(1 if classes == 2 else classes, name='predictions')(outputs)
        outputs = get_activation_from_name(final_activation)(outputs)
    else:
        if pooling == 'avg':
            outputs = GlobalAveragePooling2D(name='global_avgpool')(outputs)
        elif pooling == 'max':
            outputs = GlobalMaxPooling2D(name='global_maxpool')(outputs)
            
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    if num_blocks == [3, 4, 6, 3]:
        model = Model(inputs, outputs, name='ResNet-50')
    elif num_blocks == [3, 4, 23, 3]:
        model = Model(inputs, outputs, name='ResNet-101')
    elif num_blocks == [3, 8, 36, 3]:
        model = Model(inputs, outputs, name='ResNet-152')
    else:
        model = Model(inputs, outputs, name='ResNet-B')
    
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if num_blocks == [3, 4, 6, 3]:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    RESNET50_WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            elif num_blocks == [3, 4, 23, 3]:
                weights_path = None
            elif num_blocks == [3, 8, 36, 3]:
                weights_path = None
        else:
            if num_blocks == [3, 4, 6, 3]:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    RESNET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
            elif num_blocks == [3, 4, 23, 3]:
                weights_path = None
            elif num_blocks == [3, 8, 36, 3]:
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


def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000) -> Model:
    
    model = ResNetA(num_blocks=[2, 2, 2, 2],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def ResNet18_backbone(input_shape=(224, 224, 3),
                      include_top=False, 
                      weights='imagenet', 
                      input_tensor=None, 
                      pooling=None, 
                      final_activation="softmax",
                      classes=1000,
                      custom_layers=None) -> Model:

    model = ResNet18(include_top=include_top,
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
        return Model(inputs=model.inputs, outputs=y_i, name='ResNet18_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a2_final").output
        y_8 = model.get_layer("residual_block_b2_final").output
        y_16 = model.get_layer("residual_block_c2_final").output
        y_32 = model.get_layer("residual_block_d2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='ResNet18_backbone')


def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000) -> Model:
    
    model = ResNetA(num_blocks=[3, 4, 6, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def ResNet34_backbone(input_shape=(224, 224, 3),
                      include_top=False, 
                      weights='imagenet', 
                      input_tensor=None, 
                      pooling=None, 
                      final_activation="softmax",
                      classes=1000,
                      custom_layers=None) -> Model:

    model = ResNet34(include_top=include_top,
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
        return Model(inputs=model.inputs, outputs=y_i, name='ResNet34_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c6_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='ResNet34_backbone')


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000) -> Model:
    
    model = ResNetB(num_blocks=[3, 4, 6, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def ResNet50_backbone(input_shape=(224, 224, 3),
                      include_top=False, 
                      weights='imagenet', 
                      input_tensor=None, 
                      pooling=None, 
                      final_activation="softmax",
                      classes=1000,
                      custom_layers=None) -> Model:

    model = ResNet50(include_top=include_top,
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
        return Model(inputs=model.inputs, outputs=y_i, name='ResNet50_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c6_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='ResNet50_backbone')


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000) -> Model:
    
    model = ResNetB(num_blocks=[3, 4, 23, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def ResNet101_backbone(input_shape=(224, 224, 3),
                       include_top=False, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       final_activation="softmax",
                       classes=1000,
                       custom_layers=None) -> Model:

    model = ResNet101(include_top=include_top,
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
        return Model(inputs=model.inputs, outputs=y_i, name='ResNet101_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b4_final").output
        y_16 = model.get_layer("residual_block_c23_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='ResNet101_backbone')


def ResNet152(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000) -> Model:
    
    model = ResNetB(num_blocks=[3, 8, 36, 3],
                    include_top=include_top,
                    weights=weights, 
                    input_tensor=input_tensor, 
                    input_shape=input_shape, 
                    pooling=pooling, 
                    final_activation=final_activation,
                    classes=classes)
    return model


def ResNet152_backbone(input_shape=(224, 224, 3),
                       include_top=False, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       final_activation="softmax",
                       classes=1000,
                       custom_layers=None) -> Model:

    model = ResNet152(include_top=include_top,
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
        return Model(inputs=model.inputs, outputs=y_i, name='ResNet152_backbone')

    else:
        y_2 = model.get_layer("initial_block_maxpool").output
        y_4 = model.get_layer("residual_block_a3_final").output
        y_8 = model.get_layer("residual_block_b8_final").output
        y_16 = model.get_layer("residual_block_c36_final").output
        y_32 = model.get_layer("residual_block_d3_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='ResNet152_backbone')
