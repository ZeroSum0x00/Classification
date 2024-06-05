"""
  # Description:
    - The following table comparing the params of the DarkNet 53 with C3 Block (YOLOv5 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

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
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.utils import get_source_inputs, get_file

from .darknet53 import ConvolutionBlock
from models.layers import get_activation_from_name, get_normalizer_from_name, TransformerBlock
from utils.model_processing import _obtain_input_shape



class Contract(tf.keras.layers.Layer):
    
    """
        Contract width-height into channels.
    """
    
    def __init__(self, gain=2, axis=-1, *args, **kwargs):
        super(Contract, self).__init__(*args, **kwargs)
        self.gain = gain
        self.axis = axis

    def call(self, inputs):
        bs, h, w, c = inputs.shape
        s = self.gain
        x = tf.reshape(inputs, (-1, h // s, s, w // s, s, c))
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(inputs, (-1, h // s, w // s, c * s * s))
        return x


class Expand(tf.keras.layers.Layer):
    
    """ 
        Expand channels into width-height.
    """
    
    def __init__(self, gain=2, *args, **kwargs):
        super(Expand, self).__init__(*args, **kwargs)
        self.gain = gain

    def call(self, inputs):
        bs, h, w, c = inputs.shape
        s = self.gain
        x = tf.reshape(inputs, (-1, h, w, c // s ** 2, s, s))
        x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3])
        x = tf.reshape(inputs, (-1, h * s, w * s, c // s ** 2))
        return x


class Focus(tf.keras.layers.Layer):
    
    '''
        Focus wh information into c-space.
    '''
    
    def __init__(self,
                 filters,
                 kernel_size,
                 downsample=False,
                 groups=1,
                 activation='relu', 
                 normalizer='batch-norm',
                 *args, 
                 **kwargs):
        super(Focus, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.downsample    = downsample
        self.groups        = groups
        self.activation    = activation
        self.normalizer    = normalizer

    def build(self, input_shape):
        self.conv = ConvolutionBlock(self.filters, 
                                     self.kernel_size, 
                                     downsample=self.downsample, 
                                     groups=self.groups, 
                                     activation=self.activation, 
                                     normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, 1::2, ::2, :]
        x3 = inputs[:, ::2, 1::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = concatenate([x1, x2, x3, x4], axis=-1)  
        x = self.conv(x, training=training)
        return x


class StemBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size,
                 strides,
                 activation        = 'silu', 
                 normalizer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        self.filters           = filters
        self.kernel_size       = kernel_size
        self.strides           = strides       
        self.activation        = activation
        self.normalizer        = normalizer
        self.regularizer_decay = regularizer_decay
                     
    def build(self, input_shape):
        self.padding = ZeroPadding2D(padding=((2, 2),(2, 2)))
        self.conv    = Conv2D(filters=self.filters, 
                              kernel_size=self.kernel_size, 
                              strides=self.strides,
                              padding="valid", 
                              use_bias=not self.normalizer, 
                              kernel_initializer=RandomNormal(stddev=0.02),
                              kernel_regularizer=l2(self.regularizer_decay))
        self.norm    = get_normalizer_from_name(self.normalizer)
        self.activ   = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x = self.padding(inputs)
        x = self.conv(x, training=training)
        x = self.norm(x, training=training)
        x = self.activ(x, training=training)
        return x

        
class Bottleneck(tf.keras.layers.Layer):

    """
        Standard bottleneck.
    """
    
    def __init__(self, 
                 filters, 
                 downsample = False,
                 groups     = 1,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.filters    = filters
        self.downsample = downsample
        self.groups     = groups
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 3, downsample=self.downsample, groups=self.groups, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x

        
class BottleneckCSP(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks 
    """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion         = 0.5,
                 shortcut          = True,
                 activation        = 'silu', 
                 normalizer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(BottleneckCSP, self).__init__(**kwargs)
        self.filters           = filters
        self.iters             = iters
        self.expansion         = expansion
        self.shortcut          = shortcut       
        self.activation        = activation
        self.normalizer        = normalizer
        self.regularizer_decay = regularizer_decay
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.middle = Sequential([
            Bottleneck(hidden_dim, shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])
        self.conv2 = Conv2D(filters=hidden_dim, 
                            kernel_size=(1, 1), 
                            strides=(1, 1),
                            padding="valid", 
                            use_bias=not self.normalizer, 
                            kernel_initializer=RandomNormal(stddev=0.02),
                            kernel_regularizer=l2(self.regularizer_decay))
        self.shortcut = Conv2D(filters=hidden_dim, 
                               kernel_size=(1, 1), 
                               strides=(1, 1),
                               padding="valid", 
                               use_bias=not self.normalizer, 
                               kernel_initializer=RandomNormal(stddev=0.02),
                               kernel_regularizer=l2(self.regularizer_decay))
        self.norm = get_normalizer_from_name(self.normalizer)
        self.activ = get_activation_from_name(self.activation)
        self.conv3 = Conv2D(filters=self.filters, 
                            kernel_size=(1, 1), 
                            strides=(1, 1),
                            padding="valid", 
                            use_bias=not self.normalizer, 
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
    
    """ 
        CSP Bottleneck with 3 convolutions.
    """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(C3, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.middle = Sequential([
            Bottleneck(hidden_dim, shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])
        self.residual = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.middle(x, training=training)
        y = self.residual(inputs, training=training)
        
        merger = concatenate([x, y], axis=-1)
        merger = self.conv2(merger, training=training)
        return merger


class CrossConv2D(tf.keras.layers.Layer):
    
    """ 
        Cross Convolution Downsample.
    """
    
    def __init__(self, 
                 filters, 
                 kernel_size,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(CrossConv2D, self).__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.expansion   = expansion
        self.shortcut    = shortcut       
        self.activation  = activation
        self.normalizer  = normalizer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, (1, self.kernel_size), activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, (self.kernel_size, 1), activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class C3x(C3):
    
    """ 
        C3 module with cross-convolutions.
    """

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = Sequential([
            CrossConv2D(hidden_dim, kernel_size=3, shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])


class SPP(tf.keras.layers.Layer):
    
    """ 
        Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729 
    """
    
    def __init__(self, 
                 filters, 
                 pool_pyramid = (5, 9, 13),
                 expansion    = 0.5,
                 activation   = 'silu', 
                 normalizer   = 'batch-norm', 
                 **kwargs):
        super(SPP, self).__init__(**kwargs)
        self.filters      = filters
        self.pool_pyramid = pool_pyramid
        self.expansion    = expansion
        self.activation   = activation
        self.normalizer   = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.pool1 = MaxPooling2D(pool_size=self.pool_pyramid[0], strides=(1, 1), padding='same')
        self.pool2 = MaxPooling2D(pool_size=self.pool_pyramid[1], strides=(1, 1), padding='same')
        self.pool3 = MaxPooling2D(pool_size=self.pool_pyramid[2], strides=(1, 1), padding='same')
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        p1 = self.pool1(x)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)
        x = concatenate([x, p1, p2, p3], axis=-1)
        x = self.conv2(x, training=training)
        return x


class C3SPP(C3):
    
    """ 
    C3 module with SPP.
    """

    def __init__(self, 
                 filters, 
                 iters,
                 pool_pyramid = (5, 9, 13),
                 expansion    = 0.5,
                 activation   = 'silu', 
                 normalizer   = 'batch-norm', 
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         activation,
                         **kwargs)
        self.pool_pyramid = pool_pyramid
                     
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = Sequential([
            SPP(hidden_dim, self.pool_pyramid) for i in range(self.iters)
        ])
        

class SPPF(tf.keras.layers.Layer):
    
    """ 
        Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.
    """
    
    def __init__(self, 
                 filters, 
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(SPPF, self).__init__(**kwargs)
        self.filters    = filters
        self.pool_size  = pool_size
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(input_shape[-1] * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        x = concatenate([x, p1, p2, p3], axis=-1)
        x = self.conv2(x, training=training)
        return x


class C3SPPF(C3):
    
    """ 
        C3 module with SPP.
    """

    def __init__(self, 
                 filters, 
                 iters,
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         activation,
                         **kwargs)
        self.pool_size = pool_size
                     
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = Sequential([
            SPPF(hidden_dim, self.pool_size) for i in range(self.iters)
        ])


class GhostConv(tf.keras.layers.Layer):
    
    """ 
    Ghost Convolution https://github.com/huawei-noah/ghostnet.
    """
    
    def __init__(self, 
                 filters, 
                 kernel_size =(1, 1),
                 downsample  = False,
                 expansion   = 0.5,
                 activation  = 'silu', 
                 normalizer  = 'batch-norm', 
                 **kwargs):
        super(GhostConv, self).__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.downsample  = downsample
        self.expansion   = expansion
        self.activation  = activation
        self.normalizer  = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, self.kernel_size, downsample=self.downsample, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(hidden_dim, 5, groups=hidden_dim, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        y = self.conv2(x, training=training)
        return concatenate([x, y], axis=-1)


class GhostBottleneck(tf.keras.layers.Layer):
    
    """ 
        Ghost Convolution https://github.com/huawei-noah/ghostnet 
    """
    
    def __init__(self, 
                 filters, 
                 dwkernel   = 3,
                 stride     = 1,
                 expansion  = 0.5,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(GhostBottleneck, self).__init__(**kwargs)
        self.filters    = filters
        self.dwkernel   = dwkernel
        self.stride     = stride
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = GhostConv(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = GhostConv(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        
        if self.stride == 2:
            self.dw1 = self._depthwise_block(self.dwkernel, self.stride, self.activation, self.normalizer)
            self.dw2 = self._depthwise_block(self.dwkernel, self.stride, self.activation, self.normalizer)
            self.shortcut = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def _depthwise_block(self, dwkernel, stride, activation, normalizer):
        return Sequential([
            DepthwiseConv2D(dwkernel, 
                            stride, 
                            padding="same", 
                            use_bias=False, 
                            depthwise_initializer=VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")),
            get_normalizer_from_name(normalizer),
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
    
    """ 
        C3 module with GhostBottleneck.
    """

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = Sequential([
            GhostBottleneck(hidden_dim, dwkernel=3, stride=1, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])


class TransfomerProjection(tf.keras.layers.Layer):

    """
        Vision Transformer https://arxiv.org/abs/2010.11929
    """
    
    def __init__(self, num_heads, mlp_dim, iters=1, activation='silu', normalizer='batch-norm', norm_eps=1e-6, *args, **kwargs):
        super(TransfomerProjection, self).__init__(*args, **kwargs)
        self.num_heads  = num_heads
        self.mlp_dim    = mlp_dim
        self.iters      = iters
        self.activation = activation
        self.normalizer = normalizer
        self.norm_eps   = norm_eps

    def build(self, input_shape):
        if self.mlp_dim != input_shape[-1]:
            self.channel_project = ConvolutionBlock(self.mlp_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.position = Dense(units=self.mlp_dim)
        self.transfomer_sequence = [TransformerBlock(self.num_heads, 
                                                     self.mlp_dim, 
                                                     return_weight=False, 
                                                     activation=self.activation, 
                                                     normalizer=self.normalizer, 
                                                     norm_eps=self.norm_eps,
                                                     name=f"Transformer/encoderblock_{i}") for i in range(self.iters)]
    
    def call(self, inputs, training=False):
        if hasattr(self, 'channel_project'):
            inputs = self.channel_project(inputs, training=training)
        bs, h, w, c = inputs.shape
        x = tf.reshape(inputs, (-1, h * w, c))
        x = self.position(x, training=training)
        for transfomer in self.transfomer_sequence:
            x, _ = transfomer(x, training=training)
        x = tf.reshape(inputs, (-1, h, w, self.mlp_dim))
        return x


class C3Trans(C3):
    
    """ 
        C3 module with Vision Transfomer blocks
    """

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = TransfomerProjection(4, hidden_dim, iters=self.iters, activation=self.activation, normalizer=self.normalizer)


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
              normalizer='batch-norm',
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

    l0, l1, l2, l3 = layers
            
    x = StemBlock(filters, 6, 2, activation=activation, normalizer=normalizer, name='stem')(img_input)

    x = ConvolutionBlock(filters * 2, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage1.block1')(x)
    x = c3_block(filters * 2, l0, activation=activation, normalizer=normalizer, name='stage1.block2')(x)

    x = ConvolutionBlock(filters * 4, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage2.block1')(x)
    x = c3_block(filters * 4, l1, activation=activation, normalizer=normalizer, name='stage2.block2')(x)

    x = ConvolutionBlock(filters * 8, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage3.block1')(x)
    x = c3_block(filters * 8, l2, activation=activation, normalizer=normalizer, name='stage3.block2')(x)

    x = ConvolutionBlock(filters * 16, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage4.block1')(x)
    x = c3_block(filters * 16, l3, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    x = spp_block(filters * 16, name='stage4.block3')(x)

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
                   normalizer='batch-norm',
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
                      normalizer=normalizer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_nano_backbone(c3_block=C3,
                            spp_block=SPP,
                            input_shape=(640, 640, 3),
                            include_top=False, 
                            weights='imagenet', 
                            activation='silu',
                            normalizer='batch-norm',
                            custom_layers=None) -> Model:
    
    """
        - Used in YOLOv5 version nano
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5n.yaml
    """
    
    model = DarkNetC3_nano(c3_block=c3_block,
                           spp_block=spp_block,
                           include_top=include_top, 
                           weights=weights,
                           activation=activation,
                           normalizer=normalizer,
                           input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')
        

def DarkNetC3_small(c3_block=C3,
                    spp_block=SPP,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
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
                      normalizer=normalizer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_small_backbone(c3_block=C3,
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:

    """
        - Used in YOLOv5 version small
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml
    """
    
    model = DarkNetC3_small(c3_block=c3_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            normalizer=normalizer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_medium(c3_block=C3,
                     spp_block=SPP,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
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
                      normalizer=normalizer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_medium_backbone(c3_block=C3,
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:
    
    """
        - Used in YOLOv5 version medium
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
    """
    
    model = DarkNetC3_medium(c3_block=c3_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             normalizer=normalizer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_large(c3_block=C3,
                    spp_block=SPP,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
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
                      normalizer=normalizer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_large_backbone(c3_block=C3,
                             spp_block=SPP,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:
    
    """
        - Used in YOLOv5 version large
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5l.yaml
    """
    
    model = DarkNetC3_large(c3_block=c3_block,
                            spp_block=spp_block,
                            include_top=include_top, 
                            weights=weights,
                            activation=activation,
                            normalizer=normalizer,
                            input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetC3_xlarge(c3_block=C3,
                     spp_block=SPP,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
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
                      normalizer=normalizer,
                      final_activation=final_activation,
                      classes=classes)
    return model


def DarkNetC3_xlarge_backbone(c3_block=C3,
                              spp_block=SPP,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:

    """
        - Used in YOLOv5 version xlarge
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5x.yaml
    """

    model = DarkNetC3_xlarge(c3_block=c3_block,
                             spp_block=spp_block,
                             include_top=include_top, 
                             weights=weights,
                             activation=activation,
                             normalizer=normalizer,
                             input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stem").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')