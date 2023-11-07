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
from tensorflow.keras import Sequential
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

from .darknet53 import ConvolutionBlock
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


class StemBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size,
                 strides,
                 activation        = 'silu', 
                 norm_layer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        self.filters           = filters
        self.kernel_size       = kernel_size
        self.strides           = strides       
        self.activation        = activation
        self.norm_layer        = norm_layer
        self.regularizer_decay = regularizer_decay
                     
    def build(self, input_shape):
        self.padding = ZeroPadding2D(padding=((2, 2),(2, 2)))
        self.conv    = Conv2D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=self.strides,
                              padding="valid", 
                              use_bias=not self.norm_layer, 
                              kernel_initializer=RandomNormal(stddev=0.02),
                              kernel_regularizer=l2(self.regularizer_decay))
        self.norm    = get_normalizer_from_name(self.norm_layer)
        self.activ   = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x = self.padding(inputs)
        x = self.conv(x, training=training)
        x = self.norm(x, training=training)
        x = self.activ(x, training=training)
        return x

        
class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.filters    = filters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 3, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x

        
class BottleneckCSP(tf.keras.layers.Layer):
    """ CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion         = 0.5,
                 shortcut          = True,
                 activation        = 'silu', 
                 norm_layer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(BottleneckCSP, self).__init__(**kwargs)
        self.filters           = filters
        self.iters             = iters
        self.expansion         = expansion
        self.shortcut          = shortcut       
        self.activation        = activation
        self.norm_layer        = norm_layer
        self.regularizer_decay = regularizer_decay
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            Bottleneck(hidden_dim, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])
        self.conv2 = Conv2D(filters=hidden_dim, 
                            kernel_size=(1, 1), 
                            strides=(1, 1),
                            padding="valid", 
                            use_bias=not self.norm_layer, 
                            kernel_initializer=RandomNormal(stddev=0.02),
                            kernel_regularizer=l2(self.regularizer_decay))
        self.shortcut = Conv2D(filters=hidden_dim, 
                               kernel_size=(1, 1), 
                               strides=(1, 1),
                               padding="valid", 
                               use_bias=not self.norm_layer, 
                               kernel_initializer=RandomNormal(stddev=0.02),
                               kernel_regularizer=l2(self.regularizer_decay))
        self.norm = get_normalizer_from_name(self.norm_layer)
        self.activ = get_activation_from_name(self.activation)
        self.conv3 = Conv2D(filters=self.filters, 
                            kernel_size=(1, 1), 
                            strides=(1, 1),
                            padding="valid", 
                            use_bias=not self.norm_layer, 
                            kernel_initializer=RandomNormal(stddev=0.02),
                            kernel_regularizer=l2(self.regularizer_decay))

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.middle(x, training=training)
        x = self.conv2(x, training=training)
        y = self.shortcut(inputs, training=training)
        
        merger = concatenate([x, y], axis=-1)
        merger = self.norm(merger, training=training)
        merger = self.activ(merger, training=training)
        merger = self.conv3(merger, training=training)
        return merger

        
class C3(tf.keras.layers.Layer):
    """ CSP Bottleneck with 3 convolutions """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(C3, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            Bottleneck(hidden_dim, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])
        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.middle(x, training=training)
        y = self.shortcut(inputs, training=training)
        
        merger = concatenate([x, y], axis=-1)
        merger = self.conv2(merger, training=training)
        return merger

        
class CrossConv2D(tf.keras.layers.Layer):
    """ Cross Convolution Downsample """
    
    def __init__(self, 
                 filters, 
                 kernel_size,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(CrossConv2D, self).__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.expansion   = expansion
        self.shortcut    = shortcut       
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, (1, self.kernel_size), activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, (self.kernel_size, 1), activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class C3x(C3):
    """ C3 module with cross-convolutions """

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            CrossConv2D(hidden_dim, kernel_size=3, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])
        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)


class SPP(tf.keras.layers.Layer):
    """ Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729 """
    
    def __init__(self, 
                 filters, 
                 pool_pyramid = (5, 9, 13),
                 activation   = 'silu', 
                 norm_layer   = 'batch-norm', 
                 **kwargs):
        super(SPP, self).__init__(**kwargs)
        self.filters      = filters
        self.pool_pyramid = pool_pyramid
        self.activation   = activation
        self.norm_layer   = norm_layer
                     
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.filters // 2, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.pool1 = MaxPooling2D(pool_size=self.pool_pyramid[0], strides=(1, 1), padding='same')
        self.pool2 = MaxPooling2D(pool_size=self.pool_pyramid[1], strides=(1, 1), padding='same')
        self.pool3 = MaxPooling2D(pool_size=self.pool_pyramid[2], strides=(1, 1), padding='same')
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        p1 = self.pool1(x)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)
        x = concatenate([x, p1, p2, p3], axis=-1)
        x = self.conv2(x, training=training)
        return x


class C3SPP(C3):
    """ C3 module with SPP """

    def __init__(self, 
                 filters, 
                 iters,
                 pool_pyramid = (5, 9, 13),
                 expansion    = 0.5,
                 activation   = 'silu', 
                 norm_layer   = 'batch-norm', 
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         activation,
                         **kwargs)
        self.pool_pyramid = pool_pyramid
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            SPP(hidden_dim, self.pool_pyramid) for i in range(self.iters)
        ])
        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        

class SPPF(tf.keras.layers.Layer):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher """
    
    def __init__(self, 
                 filters, 
                 pool_size  = (5, 5),
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(SPPF, self).__init__(**kwargs)
        self.filters    = filters
        self.pool_size  = pool_size
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = input_shape[-1] // 2
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.pool1 = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.pool2 = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.pool3 = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        p1 = self.pool1(x)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)
        x = concatenate([x, p1, p2, p3], axis=-1)
        x = self.conv2(x, training=training)
        return x


class C3SPPF(C3):
    """ C3 module with SPP """

    def __init__(self, 
                 filters, 
                 iters,
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         activation,
                         **kwargs)
        self.pool_size = pool_size
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            SPPF(hidden_dim, self.pool_size) for i in range(self.iters)
        ])
        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)


class GhostConv(tf.keras.layers.Layer):
    """ Ghost Convolution https://github.com/huawei-noah/ghostnet """
    
    def __init__(self, 
                 filters, 
                 kernel_size =(1, 1),
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 **kwargs):
        super(GhostConv, self).__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters // 2)
        self.conv1 = ConvolutionBlock(hidden_dim, self.kernel_size, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(hidden_dim, 5, groups=hidden_dim, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        y = self.conv2(x, training=training)
        return concatenate([x, y], axis=-1)


class GhostBottleneck(tf.keras.layers.Layer):
    """ Ghost Convolution https://github.com/huawei-noah/ghostnet """
    
    def __init__(self, 
                 filters, 
                 dwkernel   = 3,
                 stride     = 1,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(GhostBottleneck, self).__init__(**kwargs)
        self.filters    = filters
        self.dwkernel   = dwkernel
        self.stride     = stride
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters // 2)
        self.conv1 = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = GhostConv(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        
        if self.stride == 2:
            self.dw1 = _depthwise_block(dwkernel, stride, self.activation, self.norm_layer)
            self.dw2 = _depthwise_block(dwkernel, stride, self.activation, self.norm_layer)
            self.shortcut = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

    def _depthwise_block(self, dwkernel, stride, activation, norm_layer):
        return Sequential([
            DepthwiseConv2D(dwkernel, 
                            stride, 
                            padding="same", 
                            use_bias=False, 
                            depthwise_initializer=VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")),
            get_normalizer_from_name(norm_layer),
            get_activation_from_name(activation)
        ])
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)

        if self.stride == 2:
            x = self.dw1(x, training=training)

            y = self.dw2(inputs, training=training)
            y = self.shortcut(y, training=training)
        else:
            y = inputs
            
        x = self.conv2(x, training=training)
        return add([x, y])


class C3Ghost(C3):
    """ C3 module with GhostBottleneck """

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            GhostBottleneck(hidden_dim, dwkernel=3, stride=1, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])
        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

        
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
            
    x = StemBlock(filters, 6, 2, activation=activation, norm_layer=norm_layer, name='stem')(img_input)

    x = ConvolutionBlock(filters * 2, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage1_block1')(x)
    x = c3_block(filters * 2, l0, activation=activation, norm_layer=norm_layer, name='stage1_block2')(x)

    x = ConvolutionBlock(filters * 4, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage2_block1')(x)
    x = c3_block(filters * 4, l1, activation=activation, norm_layer=norm_layer, name='stage2_block2')(x)

    x = ConvolutionBlock(filters * 8, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage3_block1')(x)
    x = c3_block(filters * 8, l2, activation=activation, norm_layer=norm_layer, name='stage3_block2')(x)

    x = ConvolutionBlock(filters * 16, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage4_block1')(x)
    x = c3_block(filters * 16, l3, activation=activation, norm_layer=norm_layer, name='stage4_block2')(x)
    x = spp_block(filters * 16, name='stage4_block3')(x)

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
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1_block2").output
        y_8 = model.get_layer("stage2_block2").output
        y_16 = model.get_layer("stage3_block2").output
        y_32 = model.get_layer("stage4_block3").output
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
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1_block2").output
        y_8 = model.get_layer("stage2_block2").output
        y_16 = model.get_layer("stage3_block2").output
        y_32 = model.get_layer("stage4_block3").output
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
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1_block2").output
        y_8 = model.get_layer("stage2_block2").output
        y_16 = model.get_layer("stage3_block2").output
        y_32 = model.get_layer("stage4_block3").output
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
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1_block2").output
        y_8 = model.get_layer("stage2_block2").output
        y_16 = model.get_layer("stage3_block2").output
        y_32 = model.get_layer("stage4_block3").output
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
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1_block2").output
        y_8 = model.get_layer("stage2_block2").output
        y_16 = model.get_layer("stage3_block2").output
        y_32 = model.get_layer("stage4_block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')