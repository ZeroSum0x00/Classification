"""
  # Description:
    - The following table comparing the params of the DarkNet ELAN (YOLOv7 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       --------------------------------------------
      |      Model Name          |    Params       |
      |--------------------------------------------|
      |    DarkNetELAN tiny      |    3,071,304    |
      |--------------------------------------------|
      |    DarkNetELAN nano      |   14,416,840    |
      |--------------------------------------------|
      |    DarkNetELAN medium    |   37,040,616    |
      |--------------------------------------------|
      |    DarkNetELAN large     |   48,830,632    |
      |--------------------------------------------|
      |    DarkNetELAN xlarge    |   66,540,136    |
      |--------------------------------------------|
      |    DarkNetELAN huge      |   84,323,624    |
       --------------------------------------------

  # Reference:
    - Source: https://github.com/WongKinYiu/yolov7

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
from .darknetc3 import Bottleneck, GhostConv, GhostBottleneck
from .efficient_rep import CSPSPPF, RepVGGBlock

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


class ReOrg(tf.keras.layers.Layer):
    def __init__(self, axis=-1, *args, **kwargs):
        super(ReOrg, self).__init__(*args, **kwargs)
        self.axis = axis

    def call(self, inputs):
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, 1::2, ::2, :]
        x3 = inputs[:, ::2, 1::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = concatenate([x1, x2, x3, x4], axis=self.axis)        
        return x


class ChunCat(tf.keras.layers.Layer):
    
    def __init__(self, chun_dim=2, axis=-1, *args, **kwargs):
        super(ChunCat, self).__init__(*args, **kwargs)
        self.chun_dim = chun_dim
        self.axis     = axis
        self.merger   = Concatenate(axis=self.axis)
        
    def call(self, inputs):
        x1 = []
        x2 = []
        for input in inputs:
            i1, i2 = tf.split(input, num_or_size_splits=[self.chun_dim, inputs.shape[self.axis] - self.chun_dim], axis=self.axis)
            x1.append(i1)
            x2.append(i2)
        x = self.merger(x1 + x2)
        return x


class Shortcut(tf.keras.layers.Layer):
    
    def __init__(self, axis=-1, *args, **kwargs):
        super(Shortcut, self).__init__(*args, **kwargs)
        self.axis = axis
        
    def call(self, inputs, shortcut=None):
        if shortcut is not None:
            return inputs + shortcut
        else:
            return inputs


class FoldCut(tf.keras.layers.Layer):
    
    def __init__(self, fold_dim=2, axis=-1, *args, **kwargs):
        super(FoldCut, self).__init__(*args, **kwargs)
        self.fold_dim = fold_dim
        self.axis     = axis
        
    def call(self, inputs):
        x1, x2 = tf.split(inputs, num_or_size_splits=[self.fold_dim, inputs.shape[self.axis] - self.fold_dim], axis=self.axis)
        return x1 + x2


class ImplicitAdd(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, stddev=0.02, *args, **kwargs):
        super(ImplicitAdd, self).__init__(*args, **kwargs)
        self.mean   = mean
        self.stddev = stddev
                     
    def build(self, input_shape):
        init_value = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        self.implicit = tf.Variable(name="implicit",
                                    initial_value=init_value(shape=(1, 1, 1, input_shape[-1]), dtype=tf.float32),
                                    trainable=True)
        
    def call(self, inputs, training=False):
        return inputs + self.implicit


class ImplicitMul(tf.keras.layers.Layer):
    def __init__(self, mean=1.0, stddev=0.02, *args, **kwargs):
        super(ImplicitMul, self).__init__(*args, **kwargs)
        self.mean   = mean
        self.stddev = stddev
                     
    def build(self, input_shape):
        init_value = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        self.implicit = tf.Variable(name="implicit",
                                    initial_value=init_value(shape=(1, 1, 1, input_shape[-1]), dtype=tf.float32),
                                    trainable=True)

    def call(self, inputs, training=False):
        return inputs * self.implicit


class RobustConv(tf.keras.layers.Layer):
    
    '''
        Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    '''
    
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="same",
                 groups=1,
                 conv_scale_init=1.0,
                 activation='relu', 
                 norm_layer='batch-norm',
                 *args, 
                 **kwargs):
        super(RobustConv, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding
        self.groups = groups
        self.conv_scale_init = conv_scale_init
        self.activation = activation
        self.norm_layer = norm_layer
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.conv_dw  = self.convolution_block(hidden_dim, 
                                               self.kernel_size,
                                               self.strides,
                                               self.padding,
                                               groups=hidden_dim)
        self.conv_1x1 = Conv2D(filters=self.filters,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               use_bias=True)
        
        if self.conv_scale_init > 0:
            self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
            
        super().build(input_shape)
        
    def convolution_block(self, filters, kernel_size, strides, padding, groups):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       groups=groups,
                       use_bias=False),
                get_normalizer_from_name(self.norm_layer),
                get_activation_from_name(self.activation)
        ])
    
    def call(self, inputs, training=False):
        x = self.conv_dw(inputs, training=training)
        x = self.conv_1x1(x, training=training)
        if hasattr(self, 'scale_layer'):
            x = self.scale_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "groups": self.groups,
            "conv_scale_init": self.conv_scale_init,
            "activation": self.activation,
            "norm_layer": self.norm_layer
        })
        return config


class RobustConv2(RobustConv):
    
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="same",
                 groups=1,
                 conv_scale_init=1.0,
                 activation='relu', 
                 norm_layer='batch-norm',
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         kernel_size,
                         strides,
                         padding,
                         groups,
                         conv_scale_init,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.conv_deconv = Conv2DTranspose(filters=self.filters,
                                           kernel_size=self.strides,
                                           strides=self.strides,
                                           padding='valid',
                                           use_bias=True)
        if self.conv_scale_init > 0:
            self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv_dw(inputs, training=training)
        x = self.conv_deconv(x, training=training)
        if hasattr(self, 'scale_layer'):
            x = self.scale_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "groups": self.groups,
            "conv_scale_init": self.conv_scale_init,
            "activation": self.activation,
            "norm_layer": self.norm_layer
        })
        return config


class BasicStem(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 pool_size         = (2, 2),
                 activation        = 'silu', 
                 norm_layer        = 'batch-norm', 
                 *args,
                 **kwargs):
        super(BasicStem, self).__init__(*args, **kwargs)
        self.filters           = filters
        self.pool_size         = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size)
        self.activation        = activation
        self.norm_layer        = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = self.filters // 2
        self.conv1 = ConvolutionBlock(hidden_dim, 3, downsample=True, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(hidden_dim, 3, downsample=True, activation=self.activation, norm_layer=self.norm_layer)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2))
        self.conv4 =  ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        outputs = []
        x= self.conv1(inputs, training=training)
        outputs.append(x)
        x1 = self.conv2(x, training=training)
        x1 = self.conv3(x1, training=training)
        x2 = self.pool(x)
        x = concatenate([x1, x2], axis=-1)
        x = self.conv4(x, training=training)
        outputs.append(x)
        return outputs


class GhostStem(BasicStem):

    def __init__(self, 
                 filters, 
                 pool_size  = (2, 2),
                 activation = 'relu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         pool_size,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)

    def build(self, input_shape):
        hidden_dim = self.filters // 2
        self.conv1 = GhostConv(hidden_dim, 3, downsample=True, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = GhostConv(hidden_dim, 3, downsample=True, activation=self.activation, norm_layer=self.norm_layer)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2))
        self.conv4 =  GhostConv(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)


class DownC(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 pool_size=(2, 2),
                 activation        = 'silu', 
                 norm_layer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(DownC, self).__init__(**kwargs)
        self.filters           = filters
        self.pool_size         = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, strides)         
        self.activation        = activation
        self.norm_layer        = norm_layer
        self.regularizer_decay = regularizer_decay
                     
    def build(self, input_shape):
        hidden_dim1 = input_shape[-1]
        hidden_dim2 = self.filters // 2
        self.conv1 = ConvolutionBlock(hidden_dim1, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = self.convolution_block(hidden_dim2, 3, self.pool_size)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=self.pool_size)
        self.conv3 = ConvolutionBlock(hidden_dim2, 1, activation=self.activation, norm_layer=self.norm_layer)
        
    def convolution_block(self, filters, kernel_size, strides=1, padding="same", groups=1):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       groups=groups,
                       use_bias=False),
                get_normalizer_from_name(self.norm_layer),
                get_activation_from_name(self.activation)
        ])
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        y = self.pool(inputs)
        y = self.conv3(y, training=training)
        out = concatenate([x, y], axis=-1)
        return out


class ResX(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 **kwargs):
        super(ResX, self).__init__(**kwargs)
        self.filters     = filters
        self.groups      = groups
        self.expansion   = expansion
        self.shortcut    = shortcut       
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(hidden_dim, 3, groups=self.groups, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)        
        x = self.conv3(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class CSPSPPC(CSPSPPF):

    def __init__(self, 
                 filters, 
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'relu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         pool_size,
                         expansion,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)

    def build(self, input_shape):
        hidden_dim = int(2 * self.filters * self.expansion)
        super().build(input_shape)


class GhostCSPSPPC(CSPSPPC):

    def __init__(self, 
                 filters, 
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'relu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         pool_size,
                         expansion,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = GhostConv(hidden_dim, 3, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.conv4 = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv5 = GhostConv(hidden_dim, 3, activation=self.activation, norm_layer=self.norm_layer)
        self.conv6 = GhostConv(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)

        self.shortcut = GhostConv(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)


class BottleneckCSPA(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(BottleneckCSPA, self).__init__(**kwargs)
        self.filters     = filters
        self.groups      = groups
        self.iters       = iters
        self.expansion   = expansion
        self.shortcut    = shortcut       
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.block = Sequential([
            Bottleneck(hidden_dim, groups=self.groups, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])

    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x1 = self.block(x1, training=training)

        x2 = self.conv2(inputs, training=training)
        
        x = concatenate([x1, x2], axis=-1)
        x = self.conv3(x, training=training)
        return x


class BottleneckCSPB(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        
        x1 = self.block(x, training=training)
        x2 = self.conv2(x, training=training)
        
        x = concatenate([x1, x2], axis=-1)
        x = self.conv3(x, training=training)
        return x


class BottleneckCSPC(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.conv3 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv4 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x1 = self.block(x1, training=training)
        x1 = self.conv3(x1, training=training)

        x2 = self.conv2(inputs, training=training)
        
        x = concatenate([x1, x2], axis=-1)
        x = self.conv4(x, training=training)
        return x


class ResCSPA(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ResXCSPA(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class GhostCSPA(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            GhostBottleneck(hidden_dim, 3, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ResCSPB(BottleneckCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ResXCSPB(BottleneckCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class GhostCSPB(BottleneckCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            GhostBottleneck(hidden_dim, 3, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ResCSPC(BottleneckCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ResXCSPC(BottleneckCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            ResX(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class GhostCSPC(BottleneckCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            GhostBottleneck(hidden_dim, 3, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class RepBottleneck(Bottleneck):
    def __init__(self, 
                 filters, 
                 downsample = False,
                 groups     = 1,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 training   = False,
                 *args,
                 **kwargs):
        super().__init__(filters, downsample, groups, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        self.conv2 = RepVGGBlock(self.filters, 3, 1, groups=self.groups, training=self.training)


class RepBottleneckCSPA(BottleneckCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepBottleneck(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepBottleneckCSPB(BottleneckCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepBottleneck(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepBottleneckCSPC(BottleneckCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepBottleneck(hidden_dim, groups=self.groups, expansion=1, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepRes(ResX):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 training    = False,
                 *args,
                 **kwargs):
        super().__init__(filters, groups, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        self.conv2 = RepVGGBlock(self.filters, 3, 1, groups=self.groups, training=self.training)


class RepResCSPA(ResCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepRes(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepResCSPB(ResCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepRes(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepResCSPC(ResCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 1,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepRes(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepResX(ResX):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm', 
                 training    = False,
                 *args,
                 **kwargs):
        super().__init__(filters, groups, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        self.conv2 = RepVGGBlock(self.filters, 3, 1, groups=self.groups, training=self.training)


class RepResXCSPA(ResXCSPA):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepResXCSPB(ResXCSPB):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 1,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class RepResXCSPC(ResXCSPC):
    def __init__(self, 
                 filters, 
                 groups      = 32,
                 iters       = 1,
                 expansion   = 0.5,
                 shortcut    = True,
                 activation  = 'silu', 
                 norm_layer  = 'batch-norm',
                 training    = False,
                 *args, 
                 **kwargs):
        super().__init__(filters, groups, iters, expansion, shortcut, activation, norm_layer, *args, **kwargs)
        self.training = training
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepResX(hidden_dim, groups=self.groups, expansion=0.5, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer, training=self.training) for i in range(self.iters)
        ])


class ScaleUpConcatBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 iters      = 1,
                 id_concat  = [-1, -3, -5, -6],
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(ScaleUpConcatBlock, self).__init__(**kwargs)
        self.in_filters, self.out_filters = filters
        self.iters      = iters
        self.id_concat  = id_concat       
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.in_filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.in_filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.blocks = [ConvolutionBlock(self.in_filters, 3, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)]
        self.conv3 = ConvolutionBlock(self.out_filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)

        x = [x1, x2]
        for block in self.blocks:
            x2 = block(x2, training=training)
            x.append(x2)

        x = concatenate([x[i] for i in self.id_concat], axis=-1)
        x = self.conv3(x, training=training)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.filters    = filters
        self.activation = activation
        self.norm_layer = norm_layer
                     
    def build(self, input_shape):
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv1 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.filters, 3, downsample=True, activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        x1 = self.pool(inputs)
        x1 = self.conv1(x1, training=training)
        
        x2 = self.conv2(inputs, training=training)
        x2 = self.conv3(x2, training=training)

        x = concatenate([x1, x2], axis=-1)
        return x


def DarkNetELAN_A(filters=[32, 64],
                  iters=2, 
                  id_concat=[-1, -2, -3, -4],
                  include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  activation='leaky-relu',
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
        
    f0, f1 = filters

    x = ConvolutionBlock(f0, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stem.block1')(img_input)
    
    x = ConvolutionBlock(f0 * 2, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage1.block1')(x)

    x = ConvolutionBlock(f0, 1, activation=activation, norm_layer=norm_layer, name='stage2.block1')(x)
    x = ScaleUpConcatBlock([f1, f0 * 2], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage2.block2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='stage3.block1')(x)
    x = ConvolutionBlock(f0 * 2, 1, activation=activation, norm_layer=norm_layer, name='stage3.block2')(x)
    x = ScaleUpConcatBlock([f1 * 2, f0 * 4], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage3.block3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='stage4.block1')(x)
    x = ConvolutionBlock(f0 * 4, 1, activation=activation, norm_layer=norm_layer, name='stage4.block2')(x)
    x = ScaleUpConcatBlock([f1 * 4, f0 * 8], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage4.block3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='stage5.block1')(x)
    x = ConvolutionBlock(f0 * 8, 1, activation=activation, norm_layer=norm_layer, name='stage5.block2')(x)
    x = ScaleUpConcatBlock([f1 * 8, f0 * 16], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage5.block3')(x)

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
    if filters == [32, 32] and iters == 2:
        model = Model(inputs, x, name='DarkNet-ELAN-tiny')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN-A')

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


def DarkNetELAN_B(filters=[32, 64],
                  iters=4, 
                  id_concat=[-1, -3, -5, -6],
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
        
    f0, f1 = filters
    
    x = ConvolutionBlock(f0, 3, activation=activation, norm_layer=norm_layer, name='stem.block1')(img_input)
    x = ConvolutionBlock(f0 * 2, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stem.block2')(x)
    x = ConvolutionBlock(f0 * 2, 3, activation=activation, norm_layer=norm_layer, name='stem.block3')(x)

    x = ConvolutionBlock(f0 * 4, 3, downsample=True, activation=activation, norm_layer=norm_layer, name='stage1.block1')(x)
    x = ScaleUpConcatBlock([f1, f0 * 8], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage1.block2')(x)

    x = TransitionBlock(f0 * 4, name='stage2.block1')(x)
    x = ScaleUpConcatBlock([f1 * 2, f0 * 16], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage2.block2')(x)

    x = TransitionBlock(f0 * 8, name='stage3.block1')(x)
    x = ScaleUpConcatBlock([f1 * 4, f0 * 32], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage3.block2')(x)

    x = TransitionBlock(f0 * 16, name='stage4.block1')(x)
    x = ScaleUpConcatBlock([f1 * 4, f0 * 32], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name='stage4.block2')(x)
    
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
    if filters == [32, 64] and iters == 4:
        model = Model(inputs, x, name='DarkNet-ELAN-nano')
    elif filters == [40, 64] and iters == 6:
        model = Model(inputs, x, name='DarkNet-ELAN-small')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN-B')

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


def DarkNetELAN_C(filters=[64, 64],
                  iters=4, 
                  id_concat=[-1, -3, -5, -6],
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
                                      default_size=1280,
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
        
    scale_f0 = [filters[0] * i for i in [2, 4, 8, 12, 16]]
    scale_f1 = [filters[1] * i for i in [1, 2, 4, 6, 8]]

    x = ReOrg(name='stem.block1')(img_input)
    x = ConvolutionBlock(filters[0], 3, activation=activation, norm_layer=norm_layer, name='stem.block2')(x)

    for i, (f0, f1) in enumerate(zip(scale_f0, scale_f1)):
        x = ConvolutionBlock(f0, 3, downsample=True, activation=activation, norm_layer=norm_layer, name=f'stage{i + 1}.block1')(x)
        x = ScaleUpConcatBlock([f1, f0], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name=f'stage{i + 1}.block2')(x)

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
    if filters == [64, 64] and iters == 4:
        model = Model(inputs, x, name='DarkNet-ELAN-medium')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN-C')

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


def DarkNetELAN_D(filters=[80, 64],
                  iters=6, 
                  id_concat=[-1, -3, -5, -7, -8],
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
                                      default_size=1280,
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
        
    scale_f0 = [filters[0] * i for i in [2, 4, 8, 12, 16]]
    scale_f1 = [filters[1] * i for i in [1, 2, 4, 6, 8]]

    x = ReOrg(name='stem.block1')(img_input)
    x = ConvolutionBlock(filters[0], 3, activation=activation, norm_layer=norm_layer, name='stem.block2')(x)

    for i, (f0, f1) in enumerate(zip(scale_f0, scale_f1)):
        x = DownC(f0, pool_size=(2, 2), name=f'stage{i + 1}.block1')(x)
        x = ScaleUpConcatBlock([f1, f0], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name=f'stage{i + 1}.block2')(x)

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
    if filters == [80, 64] and iters == 6:
        model = Model(inputs, x, name='DarkNet-ELAN-large')
    elif filters == [96, 64] and iters == 8:
        model = Model(inputs, x, name='DarkNet-ELAN-xlarge')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN-D')

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


def DarkNetELAN_E(filters=[80, 64],
                  iters=6, 
                  id_concat=[-1, -3, -5, -7, -8],
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
                                      default_size=1280,
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
        
    scale_f0 = [filters[0] * i for i in [2, 4, 8, 12, 16]]
    scale_f1 = [filters[1] * i for i in [1, 2, 4, 6, 8]]

    x = ReOrg(name='stem.block1')(img_input)
    x = ConvolutionBlock(filters[0], 3, activation=activation, norm_layer=norm_layer, name='stem.block2')(x)

    for i, (f0, f1) in enumerate(zip(scale_f0, scale_f1)):
        x = DownC(f0, pool_size=(2, 2), name=f'stage{i + 1}.block1')(x)
        x1 = ScaleUpConcatBlock([f1, f0], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name=f'stage{i + 1}.block2')(x)
        x2 = ScaleUpConcatBlock([f1, f0], iters=iters, id_concat=id_concat, activation=activation, norm_layer=norm_layer, name=f'stage{i + 1}.block3')(x1)
        x = Shortcut(name=f'stage{i + 1}.residual')(x1, x2)

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
    if filters == [80, 64] and iters == 6:
        model = Model(inputs, x, name='DarkNet-ELAN-huge')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN-E')

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


def DarkNetELAN_tiny(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation="leaky-relu",
                     norm_layer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetELAN_A(filters=[32, 32],
                          iters=2, 
                          id_concat=[-1, -2, -3, -4],
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


def DarkNetELAN_tiny_backbone(input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='leaky-relu',
                              norm_layer='batch-norm',
                              custom_layers=None) -> Model:

    """
        - Used in YOLOv7 tiny
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-tiny.yaml
    """
    
    model = DarkNetELAN_tiny(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block1").output
        y_4 = model.get_layer("stage2.block2").output
        y_8 = model.get_layer("stage3.block3").output
        y_16 = model.get_layer("stage4.block3").output
        y_32 = model.get_layer("stage5.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetELAN_nano(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     norm_layer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetELAN_B(filters=[32, 64],
                          iters=4, 
                          id_concat=[-1, -3, -5, -6],
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


def DarkNetELAN_nano_backbone(input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='leaky-relu',
                              norm_layer='batch-norm',
                              custom_layers=None) -> Model:

    """
        - Used in YOLOv7
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7.yaml
    """
    
    model = DarkNetELAN_nano(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block3").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetELAN_small(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      norm_layer='batch-norm',
                      final_activation="softmax",
                      classes=1000) -> Model:
    
    model = DarkNetELAN_B(filters=[40, 64],
                          iters=6, 
                          id_concat=[-1, -3, -5, -7, -8],
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


def DarkNetELAN_small_backbone(input_shape=(640, 640, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='leaky-relu',
                               norm_layer='batch-norm',
                               custom_layers=None) -> Model:

    """
        - Used in YOLOv7X
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7x.yaml
    """
    
    model = DarkNetELAN_small(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block3").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetELAN_medium(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       norm_layer='batch-norm',
                       final_activation="softmax",
                       classes=1000) -> Model:
    
    model = DarkNetELAN_C(filters=[64, 64],
                          iters=4, 
                          id_concat=[-1, -3, -5, -6],
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


def DarkNetELAN_medium_backbone(input_shape=(1280, 1280, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='leaky-relu',
                                norm_layer='batch-norm',
                                custom_layers=None) -> Model:

    """
        - Used in YOLOv7-W6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-w6.yaml
    """
    
    model = DarkNetELAN_medium(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block2").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def DarkNetELAN_large(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      norm_layer='batch-norm',
                      final_activation="softmax",
                      classes=1000) -> Model:
    
    model = DarkNetELAN_D(filters=[80, 64],
                          iters=6, 
                          id_concat=[-1, -3, -5, -7, -8],
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


def DarkNetELAN_large_backbone(input_shape=(1280, 1280, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='leaky-relu',
                               norm_layer='batch-norm',
                               custom_layers=None) -> Model:

    """
        - Used in YOLOv7-E6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-e6.yaml
    """
    
    model = DarkNetELAN_large(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block2").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def DarkNetELAN_xlarge(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       norm_layer='batch-norm',
                       final_activation="softmax",
                       classes=1000) -> Model:
    
    model = DarkNetELAN_D(filters=[96, 64],
                          iters=8, 
                          id_concat=[-1, -3, -5, -7, -9, -10],
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


def DarkNetELAN_xlarge_backbone(input_shape=(1280, 1280, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='leaky-relu',
                                norm_layer='batch-norm',
                                custom_layers=None) -> Model:

    """
        - Used in YOLOv7-D6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-d6.yaml
    """
    
    model = DarkNetELAN_xlarge(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block2").output
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def DarkNetELAN_huge(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     norm_layer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetELAN_E(filters=[80, 64],
                          iters=6, 
                          id_concat=[-1, -3, -5, -7, -8],
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


def DarkNetELAN_huge_backbone(input_shape=(1280, 1280, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='leaky-relu',
                              norm_layer='batch-norm',
                              custom_layers=None) -> Model:

    """
        - Used in YOLOv7-E6E
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-e6e.yaml
    """
    
    model = DarkNetELAN_huge(include_top=include_top, 
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
        y_2 = model.get_layer("stem.block2").output
        y_4 = model.get_layer("stage1.residual").output
        y_8 = model.get_layer("stage2.residual").output
        y_16 = model.get_layer("stage3.residual").output
        y_32 = model.get_layer("stage4.residual").output
        y_64 = model.get_layer("stage5.residual").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')
