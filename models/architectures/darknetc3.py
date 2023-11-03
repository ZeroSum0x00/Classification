"""
  # Description:
    - The following table comparing the params of the DarkNet 53 with C3 Block (in YOLOv5) in Tensorflow on 
    size 640 x 640 x 3:

       ----------------------------------------
      |      Model Name      |    Params       |
      |----------------------------------------|
      |    DarkNetC3 nano    |    1,308,648    |
      |----------------------------------------|
      |    DarkNetC3 small   |    4,695,016    |
      |----------------------------------------|
      |    DarkNetC3 medium  |   12,957,544    |
      |----------------------------------------|
      |    DarkNetC3 large   |   27,641,832    |
      |----------------------------------------|
      |    DarkNetC3 xlarge  |   50,606,440    |
       ----------------------------------------

  # Reference:
    - Source: https://github.com/ultralytics/yolov5

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import VarianceScaling

from .darknet53 import convolutional_block
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def stem(x, filters, kernel_size, strides, activation='silu', norm_layer='batch-norm', regularizer_decay=5e-4, name=None):
    x = ZeroPadding2D(padding=((2, 2),(2, 2)), name=name + "_padding")(x)
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               strides=strides,
               padding="valid", 
               use_bias=not norm_layer, 
               kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(regularizer_decay),
               name=name + "_conv")(x)
    x = get_normalizer_from_name(norm_layer, name=name + "_norm")(x)
    x = get_activation_from_name(activation, name=name + "_activ")(x)
    return x


def Bottleneck(inputs, filters, expansion=1, shortcut=True, activation='silu', norm_layer='batch-norm', name=None):
    hidden_dim = filters * expansion
    input_channel = inputs.shape[-1]
    x = convolutional_block(inputs, hidden_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    x = convolutional_block(x, filters, 3, activation=activation, norm_layer=norm_layer, name=name + '_conv2')
    
    if shortcut and input_channel == filters:
        x = add([inputs, x], name=name + '_residual')
        
    return x


def BottleneckCSP(inputs, filters, iters, expansion=1, shortcut=True, activation='silu', norm_layer='batch-norm', regularizer_decay=5e-4, name=None):
    """ CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks """
    hidden_dim = filters * expansion
    x = convolutional_block(inputs, hidden_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    
    for i in range(iters):
        x = Bottleneck(x, hidden_dim, shortcut=shortcut, activation=activation, norm_layer=norm_layer, name=name + f'_bottleneck{i + 1}')
        
    x = Conv2D(filters=hidden_dim, 
               kernel_size=(1, 1), 
               strides=(1, 1),
               padding="valid", 
               use_bias=not norm_layer, 
               kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(regularizer_decay),
               name=name + '_conv2')(x)

    y = Conv2D(filters=hidden_dim, 
               kernel_size=(1, 1), 
               strides=(1, 1),
               padding="valid", 
               use_bias=not norm_layer, 
               kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(regularizer_decay),
               name=name + '_conv3')(inputs)

    o = concatenate([x, y], name=name + '_merger')
    o = get_normalizer_from_name(norm_layer, name=name + '_norm')(o)
    o = get_activation_from_name(activation, name=name + '_activ')(o)
    o = Conv2D(filters=filters, 
               kernel_size=(1, 1), 
               strides=(1, 1),
               padding="valid", 
               use_bias=not norm_layer, 
               kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(regularizer_decay),
               name=name + '_projection')(o)
    return o

    
def C3(inputs, filters, iters, expansion=0.5, shortcut=True, activation='silu', norm_layer='batch-norm', name=None):
    # CSP Bottleneck with 3 convolutions
    hidden_channels = int(filters * expansion)
    x = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')

    for i in range(iters):
        x = Bottleneck(x, hidden_channels, shortcut=shortcut, activation=activation, norm_layer=norm_layer, name=name + f'_bottleneck{i + 1}')

    y = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')

    merger = concatenate([x, y], axis=-1, name=name + '_merger')
    merger = convolutional_block(merger, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return merger


def CrossConv2D(inputs, filters, kernel_size, expansion=1, shortcut=True, activation='silu', norm_layer='batch-norm', regularizer_decay=5e-4, name=None):
    """ Cross Convolution Downsample """
    hidden_channels = int(filters * expansion)
    input_channel = inputs.shape[-1]
    x = convolutional_block(inputs, hidden_channels, (1, kernel_size), activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    x = convolutional_block(x, filters, (kernel_size, 1), activation=activation, norm_layer=norm_layer, name=name + '_conv2')
    
    if shortcut and input_channel == filters:
        return add([inputs, x], name=name + '_residual')
    else: 
        return x

    
def C3x(inputs, filters, iters, expansion=0.5, shortcut=True, activation='silu', norm_layer='batch-norm', name=None):
    """ C3 module with cross-convolutions """
    hidden_channels = int(filters * expansion)
    x = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')

    for i in range(iters):
        x = CrossConv2D(x, hidden_channels, kernel_size=3, shortcut=shortcut, activation=activation, norm_layer=norm_layer, name=name + f'_crossconv{i + 1}')

    y = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')

    merger = concatenate([x, y], axis=-1, name=name + '_merger')
    merger = convolutional_block(merger, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return merger

    
def SPP(inputs, out_dim, pool_pyramid=(5, 9, 13), activation='silu', norm_layer='batch-norm', name=None):
    """ Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729 """
    x = convolutional_block(inputs, out_dim // 2, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv')

    pool1 = MaxPooling2D(pool_size=pool_pyramid[0], strides=(1, 1), padding='same', name=name + '_pool1')(x)
    pool2 = MaxPooling2D(pool_size=pool_pyramid[1], strides=(1, 1), padding='same', name=name + '_pool2')(pool1)
    pool3 = MaxPooling2D(pool_size=pool_pyramid[2], strides=(1, 1), padding='same', name=name + '_pool3')(pool2)

    x = concatenate([x, pool1, pool2, pool3], axis=-1, name=name + '_merger')
    x = convolutional_block(x, out_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return x


def C3SPP(inputs, filters, iters, pool_pyramid=(5, 9, 13), expansion=0.5, shortcut=True, activation='silu', norm_layer='batch-norm', name=None):
    """ C3 module with SPP """
    hidden_channels = int(filters * expansion)
    x = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    
    for i in range(iters):
        x = SPP(x, hidden_channels, pool_pyramid, name=name + f'_spp{i + 1}')
        
    y = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')

    merger = concatenate([x, y], axis=-1, name=name + '_merger')
    merger = convolutional_block(merger, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return merger


def SPPF(inputs, out_dim, pool_size=(5, 5), activation='silu', norm_layer='batch-norm', name=None):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher """
    hidden_dim = inputs.shape[-1] // 2
    x = convolutional_block(inputs, hidden_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv')
    y = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same', name=name + '_pool1')(x)
    z = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same', name=name + '_pool2')(y)
    t = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same', name=name + '_pool3')(z)

    out = concatenate([x, y, z, t], axis=-1, name=name + '_merger')
    out = convolutional_block(out, out_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return out


def C3SPPF(inputs, filters, iters, pool_size=(5, 5), expansion=0.5, shortcut=True, activation='silu', norm_layer='batch-norm', name=None):
    """ C3 module with SPPF """
    hidden_channels = int(filters * expansion)
    x = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    
    for i in range(iters):
        x = SPPF(x, hidden_channels, pool_size, name=name + f'_sppf{i + 1}')

    y = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')

    merger = concatenate([x, y], axis=-1, name=name + '_merger')
    merger = convolutional_block(merger, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return merger

    
def GhostConv(inputs, filters, kernel_size=1, activation='silu', norm_layer='batch-norm', name=None):
    """ Ghost Convolution https://github.com/huawei-noah/ghostnet """
    hidden_dim = filters // 2
    x = convolutional_block(inputs, hidden_dim, kernel_size, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    y = convolutional_block(x, hidden_dim, 5, groups=hidden_dim, activation=activation, norm_layer=norm_layer, name=name + '_conv2')
    return concatenate([x, y], axis=-1, name=name + '_merger')


def GhostBottleneck(inputs, filters, dwkernel=3, stride=1, activation='silu', norm_layer='batch-norm', name=None):
    """ Ghost Convolution https://github.com/huawei-noah/ghostnet """
    hidden_dim = filters // 2
    x = GhostConv(inputs, hidden_dim, 1, activation=activation, norm_layer=norm_layer, name=name + '_ghost1')
    if stride == 2:
        x = DepthwiseConv2D(dwkernel, 
                            stride, 
                            padding="same", 
                            use_bias=False, 
                            depthwise_initializer=VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal"),
                            name=name + '/dw1')(x)
        x = get_normalizer_from_name(norm_layer, name=name + '/norm1')(x)
        x = get_activation_from_name(activation, name=name + '/activ1')(x)
    x = GhostConv(x, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_ghost2')

    if stride == 2:
        y = DepthwiseConv2D(dwkernel, 
                            stride, 
                            padding="same", 
                            use_bias=False, 
                            depthwise_initializer=VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal"),
                            name=name + '/dw2')(inputs)
        y = get_normalizer_from_name(norm_layer, name=name + '/norm2')(y)
        y = get_activation_from_name(activation, name=name + '/activ2')(y)
        y = convolutional_block(y, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    else:
        y = inputs
        
    return add([x, y], name=name + '_merger')


def C3Ghost(inputs, filters, iters, expansion=0.5, activation='silu', norm_layer='batch-norm', name=None):
    """ C3 module with GhostBottleneck """
    hidden_channels = int(filters * expansion)
    x = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')

    for i in range(iters):
        x = GhostBottleneck(x, hidden_channels, activation=activation, norm_layer=norm_layer, name=name + f'_goshbottleneck{i + 1}')

    y = convolutional_block(inputs, hidden_channels, 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')

    merger = concatenate([x, y], axis=-1, name=name + '_merger')
    merger = convolutional_block(merger, filters, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return merger


def DarkNetC3(c3_block,
              spp_block,
              layers,
              filters,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              activation='silu',
              norm_layer='batch-norm',
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
                                      default_size=640,
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
        
    l0, l1, l2, l3 = layers
            
    x = stem(img_input, filters, 6, 2, activation=activation, norm_layer=norm_layer, name='stem')

    x = convolutional_block(x, filters * 2, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage1_block1')
    x = c3_block(x, filters * 2, l0, activation=activation, norm_layer=norm_layer, name='stage1_block2')

    x = convolutional_block(x, filters * 4, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage2_block1')
    x = c3_block(x, filters * 4, l1, activation=activation, norm_layer=norm_layer, name='stage2_block2')

    x = convolutional_block(x, filters * 8, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage3_block1')
    x = c3_block(x, filters * 8, l2, activation=activation, norm_layer=norm_layer, name='stage3_block2')

    x = convolutional_block(x, filters * 16, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage4_block1')
    x = c3_block(x, filters * 16, l3, activation=activation, norm_layer=norm_layer, name='stage4_block2')
    x = spp_block(x, filters * 16, name='stage4_block3')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
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
    if layers == [1, 2, 3, 1] and filters == 16:
        model = Model(inputs, x, name='DarkNet-C3-Nano')
    elif layers == [1, 2, 3, 1] and filters == 32:
        model = Model(inputs, x, name='DarkNet-C3-Small')
    elif layers == [2, 4, 6, 2] and filters == 48:
        model = Model(inputs, x, name='DarkNet-C3-Medium')
    elif layers == [3, 6, 9, 3] and filters == 64:
        model = Model(inputs, x, name='DarkNet-C3-Large')
    elif layers == [4, 8, 12, 4] and filters == 80:
        model = Model(inputs, x, name='DarkNet-C3-XLarge')
    else:
        model = Model(inputs, x, name='DarkNet-C3')

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


def DarkNetC3_nano(c3_block=C3,
                   spp_block=SPP,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   activation='silu',
                   norm_layer='batch-norm',
                   final_activation="softmax",
                   classes=1000) -> Model:
    
    model = DarkNetC3(c3_block=c3_block,
                      spp_block=spp_block,
                      layers=[1, 2, 3, 1],
                      filters=16,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      activation=activation,
                      norm_layer=norm_layer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_nano_backbone(c3_block=C3,
                            spp_block=SPP,
                            input_shape=(640, 640, 3),
                            include_top=False, 
                            weights='imagenet', 
                            activation='silu',
                            norm_layer='batch-norm',
                            custom_layers=None) -> Model:

    model = DarkNetC3_nano(c3_block=c3_block,
                           spp_block=spp_block,
                           include_top=include_top, 
                           weights=weights,
                           activation=activation,
                           norm_layer=norm_layer,
                           input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem_activ").output
        y_4 = model.get_layer("stage1_block2_projection/activ").output
        y_8 = model.get_layer("stage2_block2_projection/activ").output
        y_16 = model.get_layer("stage3_block2_projection/activ").output
        y_32 = model.get_layer("stage4_block3_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')
        

def DarkNetC3_small(c3_block=C3,
                    spp_block=SPP,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    norm_layer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:
    
    model = DarkNetC3(c3_block=c3_block,
                      spp_block=spp_block,
                      layers=[1, 2, 3, 1],
                      filters=32,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      activation=activation,
                      norm_layer=norm_layer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_small_backbone(c3_block=C3,
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             norm_layer='batch-norm',
                             custom_layers=None) -> Model:

    model = DarkNetC3_small(c3_block=c3_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            norm_layer=norm_layer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem_activ").output
        y_4 = model.get_layer("stage1_block2_projection/activ").output
        y_8 = model.get_layer("stage2_block2_projection/activ").output
        y_16 = model.get_layer("stage3_block2_projection/activ").output
        y_32 = model.get_layer("stage4_block3_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_medium(c3_block=C3,
                     spp_block=SPP,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     norm_layer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetC3(c3_block=c3_block,
                      spp_block=spp_block,
                      layers=[2, 4, 6, 2],
                      filters=48,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      activation=activation,
                      norm_layer=norm_layer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_medium_backbone(c3_block=C3,
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              norm_layer='batch-norm',
                              custom_layers=None) -> Model:

    model = DarkNetC3_medium(c3_block=c3_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             norm_layer=norm_layer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem_activ").output
        y_4 = model.get_layer("stage1_block2_projection/activ").output
        y_8 = model.get_layer("stage2_block2_projection/activ").output
        y_16 = model.get_layer("stage3_block2_projection/activ").output
        y_32 = model.get_layer("stage4_block3_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_large(c3_block=C3,
                    spp_block=SPP,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    norm_layer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:
    
    model = DarkNetC3(c3_block=c3_block,
                      spp_block=spp_block,
                      layers=[3, 6, 9, 3],
                      filters=64,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      activation=activation,
                      norm_layer=norm_layer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_large_backbone(c3_block=C3,
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             norm_layer='batch-norm',
                             custom_layers=None) -> Model:

    model = DarkNetC3_large(c3_block=c3_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            norm_layer=norm_layer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem_activ").output
        y_4 = model.get_layer("stage1_block2_projection/activ").output
        y_8 = model.get_layer("stage2_block2_projection/activ").output
        y_16 = model.get_layer("stage3_block2_projection/activ").output
        y_32 = model.get_layer("stage4_block3_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_xlarge(c3_block=C3,
                     spp_block=SPP,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     norm_layer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetC3(c3_block=c3_block,
                      spp_block=spp_block,
                      layers=[4, 8, 12, 4],
                      filters=80,
                      include_top=include_top,
                      weights=weights, 
                      input_tensor=input_tensor, 
                      input_shape=input_shape, 
                      pooling=pooling, 
                      activation=activation,
                      norm_layer=norm_layer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_xlarge_backbone(c3_block=C3,
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              norm_layer='batch-norm',
                              custom_layers=None) -> Model:

    model = DarkNetC3_xlarge(c3_block=c3_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             norm_layer=norm_layer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem_activ").output
        y_4 = model.get_layer("stage1_block2_projection/activ").output
        y_8 = model.get_layer("stage2_block2_projection/activ").output
        y_16 = model.get_layer("stage3_block2_projection/activ").output
        y_32 = model.get_layer("stage4_block3_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')