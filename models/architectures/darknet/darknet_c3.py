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

import copy
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, DepthwiseConv2D, MaxPooling2D, Dense, Dropout,
    GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate, add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import VarianceScaling

from .darknet53 import ConvolutionBlock
from models.layers import (
    get_activation_from_name, get_normalizer_from_name, LinearLayer,
    MultiHeadSelfAttention, MLPBlock, TransformerBlock
)
from utils.model_processing import process_model_input, create_layer_instance



class Contract(tf.keras.layers.Layer):
    
    """
        Contract width-height into channels.
    """
    
    def __init__(self, gain=2, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        super().__init__(*args, **kwargs)
        self.gain = gain

    def call(self, inputs):
        bs, h, w, c = inputs.shape
        s = self.gain
        x = tf.reshape(inputs, (-1, h, w, c // s ** 2, s, s))
        x = tf.transpose(x, perm=[0, 1, 4, 2, 5, 3])
        x = tf.reshape(inputs, (-1, h * s, w * s, c // s ** 2))
        return x


class Focus(tf.keras.layers.Layer):
    
    """
        Focus wh information into c-space.
    """
    
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        groups=1,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.conv = ConvolutionBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

    def call(self, inputs, training=False):
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, 1::2, ::2, :]
        x3 = inputs[:, ::2, 1::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = concatenate([x1, x2, x3, x4], axis=-1)  
        x = self.conv(x, training=training)
        return x


class StemBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
                     
    def build(self, input_shape):
        self.padding = ZeroPadding2D(padding=((2, 2),(2, 2)))
        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="valid",
            use_bias=not self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.norm = get_normalizer_from_name(self.normalizer)
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x = self.padding(inputs)
        x = self.conv(x, training=training)
        x = self.norm(x, training=training)
        x = self.activ(x, training=training)
        return x

        
class Bottleneck(tf.keras.layers.Layer):
    """
    Standard bottleneck layer.
    """

    def __init__(
        self,
        filters,
        kernels=(1, 3),
        strides=(1, 1),
        groups=1,
        expansion=1,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.groups = groups
        self.expansion = expansion
        self.shortcut = shortcut
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.c = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)

        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=self.kernels[0],
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=self.kernels[1],
            strides=self.strides,
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        if self.shortcut and self.c != self.filters:
            self.shortcut_layer = ConvolutionBlock(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=None,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )

    def call(self, inputs, training=False):
        residue = inputs
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if self.shortcut:
            if hasattr(self, "shortcut_layer"):
                residue = self.shortcut_layer(residue, training=training)

            x = add([x, residue])

        return x

        
class BottleneckCSP(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks 
    """
    
    def __init__(
        self,
        filters,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.iters = iters
        self.expansion = expansion
        self.shortcut = shortcut       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.middle = Sequential([
            Bottleneck(
                filters=hidden_dim,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])
        
        self.conv2 = self.convolution_block(hidden_dim)
        self.shortcut = self.convolution_block(hidden_dim)
        
        self.norm = get_normalizer_from_name(self.normalizer)
        self.activ = get_activation_from_name(self.activation)
        
        self.conv3 = self.convolution_block(self.filters)

    def convolution_block(self, filters):
        return Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=not self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )

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
    
    def __init__(
        self,
        filters,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.iters = iters
        self.expansion = expansion
        self.shortcut = shortcut       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.middle = Sequential([
            Bottleneck(
                filters=hidden_dim,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])
        
        self.residual = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

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
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        expansion=1,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.expansion = expansion
        self.shortcut = shortcut       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, self.kernel_size[1]),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(self.kernel_size[0], 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        if self.shortcut and self.c != self.filters:
            self.shortcut_layer = ConvolutionBlock(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=None,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )

    def call(self, inputs, training=False):
        residue = inputs
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut:
            if hasattr(self, "shortcut_layer"):
                residue = self.shortcut_layer(residue, training=training)

            x = add([x, residue])

        return x


class C3x(C3):
    
    """ 
        C3 module with cross-convolutions.
    """

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.middle = Sequential([
            CrossConv2D(
                filters=hidden_dim,
                kernel_size=(3, 3),
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])



class SPP(tf.keras.layers.Layer):
    
    """ 
        Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729 
    """
    
    def __init__(
        self,
        filters,
        pool_pyramid=(5, 9, 13),
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.pool_pyramid = pool_pyramid
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pool1 = MaxPooling2D(pool_size=self.pool_pyramid[0], strides=(1, 1), padding="same")
        self.pool2 = MaxPooling2D(pool_size=self.pool_pyramid[1], strides=(1, 1), padding="same")
        self.pool3 = MaxPooling2D(pool_size=self.pool_pyramid[2], strides=(1, 1), padding="same")

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

    def __init__(
        self,
        filters,
        iters=1,
        pool_pyramid=(5, 9, 13),
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            filters=filters,
            iters=iters,
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.pool_pyramid = pool_pyramid
                     
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.middle = Sequential([
            SPP(
                filters=hidden_dim,
                pool_pyramid=self.pool_pyramid,
                expansion=self.expansion,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])
        

class SPPF(tf.keras.layers.Layer):
    
    """ 
        Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.
    """
    
    def __init__(
        self,
        filters,
        pool_size=(5, 5),
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.pool_size = pool_size
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(input_shape[-1] * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding="same")

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

    def __init__(
        self,
        filters,
        iters=1,
        pool_size=(5, 5),
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            filters=filters,
            iters=iters,
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.pool_size = pool_size
                     
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)

        self.middle = Sequential([
            SPPF(
                filters=hidden_dim,
                pool_size=self.pool_size,
                expansion=self.expansion,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])

class GhostConv(tf.keras.layers.Layer):
    
    """ 
    Ghost Convolution https://github.com/huawei-noah/ghostnet.
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        expansion=0.5,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=self.kernel_size,
            strides=self.strides,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        self.conv2 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(5, 5),
            strides=(1, 1),
            groups=hidden_dim,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        y = self.conv2(x, training=training)
        return concatenate([x, y], axis=-1)


class GhostBottleneck(tf.keras.layers.Layer):
    
    """ 
        Ghost Convolution https://github.com/huawei-noah/ghostnet 
    """
    
    def __init__(
        self,
        filters,
        dwkernel=(3, 3),
        dwstride=(1, 1),
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.dwkernel = dwkernel
        self.dwstride = dwstride
        self.expansion = expansion
        self.shortcut = shortcut
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.c = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = GhostConv(
            filters=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = GhostConv(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        if self.dwstride == 2:
            self.dw1 = self._depthwise_block(self.dwkernel, self.dwstride, self.activation, self.normalizer)
            self.dw2 = self._depthwise_block(self.dwkernel, self.dwstride, self.activation, self.normalizer)

        if self.shortcut and self.c != self.filters:
            self.shortcut_layer = ConvolutionBlock(
                filters=self.filters,
                kernel_size=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )

    def _depthwise_block(self, dwkernel, strides, activation, normalizer):
        return Sequential([
            DepthwiseConv2D(
                kernel_size=dwkernel,
                strides=strides,
                padding="same",
                use_bias=False,
                depthwise_initializer=VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
            ),
            get_normalizer_from_name(normalizer),
            get_activation_from_name(activation)
        ])
        
    def call(self, inputs, training=False):
        residue = inputs
        x = self.conv1(inputs, training=training)
        
        if self.dwstride == 2:
            residue = self.dw2(residue, training=training)
            x = self.dw1(x, training=training)
            
        if hasattr(self, "shortcut_layer"):
            residue = self.shortcut_layer(residue, training=training)
            
        x = self.conv2(x, training=training)
        return add([x, residue])


class C3Ghost(C3):
    
    """ 
        C3 module with GhostBottleneck.
    """

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.middle = Sequential([
            GhostBottleneck(
                filters=hidden_dim,
                dwkernel=(3, 3),
                dwstride=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ])


class TransfomerProjection(tf.keras.layers.Layer):

    """
        Vision Transformer https://arxiv.org/abs/2010.11929
    """
    
    def __init__(
        self,
        attention_block=None,
        mlp_block=None,
        num_heads=12,
        mlp_dim=768,
        iters=1,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_rate=0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attention_block = attention_block
        self.mlp_block = mlp_block
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.iters = iters
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        if self.mlp_dim != input_shape[-1]:
            self.channel_project = ConvolutionBlock(
                filters=self.mlp_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
        
        self.position = Dense(
            units=self.mlp_dim,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )

        if self.attention_block is None:
            attn_list = [
                MultiHeadSelfAttention(
                    num_heads=self.num_heads,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    name=f"multi_head_atention{i}"
                )
                for i in range(self.iters)
            ]
        else:
            attn_list = [copy.deepcopy(self.attention_block) for i in range(self.iters)]
            
        if self.mlp_block is None:
            mlp_list = [
                MLPBlock(
                    self.mlp_dim,
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                    drop_rate=self.drop_rate,
                    name=f"mlp_block{i}")
                for i in range(self.iters)
            ]
        else:
            mlp_clone = [copy.deepcopy(self.mlp_block) for i in range(self.iters)]

        self.transfomer_sequence = [
            TransformerBlock(
                attention_block=attn_list[i],
                mlp_block=mlp_list[i],
                activation=self.activation,
                normalizer=self.normalizer,
                norm_eps=self.norm_eps,
                drop_rate=self.drop_rate,
                name=f"transformer.encoder_block{i}"
            )
            for i in range(self.iters)
        ]
    
    def call(self, inputs, training=False):
        if hasattr(self, "channel_project"):
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
        
        self.middle = TransfomerProjection(
            num_heads=4,
            mlp_dim=hidden_dim,
            iters=self.iters,
            activation=self.activation,
            normalizer=self.normalizer,
        )


def DarkNetC3(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                                   "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
    }
    
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=640,
        min_size=32,
        weights=weights
    )
    
    if isinstance(feature_extractor, (tuple, list)):
       extractor_block1, extractor_block2 = feature_extractor
    else:
        extractor_block1 = extractor_block2 = feature_extractor

    if isinstance(fusion_layer, (list, tuple)):
        fusion_block1, fusion_block2 = fusion_layer
    else:
        fusion_block1 = fusion_block2 = fusion_layer

    if pyramid_pooling and not isinstance(pyramid_pooling, (list, tuple)):
        pyramid_pooling = [pyramid_pooling]

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks))]
    
    x = inputs
    for i in range(num_blocks[0]):
        x = StemBlock(
            filters=filters[0],
            kernel_size=(6, 6),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    for i in range(len(num_blocks) - 1):
        f = filters[i + 1]
        
        x = create_layer_instance(
            extractor_block1 if i == 0 else extractor_block2,
            filters=int(f * final_channel_scale) if i == len(num_blocks) - 2 else f,
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
    
        x = create_layer_instance(
            fusion_block1 if i == 0 else fusion_block2,
            filters=int(f * final_channel_scale) if i == len(num_blocks) - 2 else f,
            iters=num_blocks[i + 1],
            **layer_constant_dict,
            name=f"stage{i + 1}.block2"
        )(x)
        
    if pyramid_pooling:
        for j, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters[-1] * final_channel_scale),
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"stage{i + 1}.block3")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    if filters == [16, 32, 64, 128, 256] and num_blocks == [1, 1, 2, 3, 1]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3-Nano")
    elif filters == [32, 64, 128, 256, 512] and num_blocks == [1, 1, 2, 3, 1]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3-Small")
    elif filters == [48, 96, 192, 384, 768] and num_blocks == [1, 2, 4, 6, 2]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3-Medium")
    elif filters == [64, 128, 256, 512, 1024] and num_blocks == [1, 3, 6, 9, 3]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3-Large")
    elif filters == [80, 160, 320, 640, 1280] and num_blocks == [1, 4, 8, 12, 4]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3-XLarge")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-C3")
        
    return model


def DarkNetC3_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetC3_nano(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C3,
        pyramid_pooling=SPP,
        filters=16,
        num_blocks=[1, 1, 2, 3, 1],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DarkNetC3_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv5 version nano
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5n.yaml
    """
    
    model = DarkNetC3_nano(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetC3_small(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C3,
        pyramid_pooling=SPP,
        filters=32,
        num_blocks=[1, 1, 2, 3, 1],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DarkNetC3_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv5 version small
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml
    """
    
    model = DarkNetC3_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

        
def DarkNetC3_medium(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C3,
        pyramid_pooling=SPP,
        filters=48,
        num_blocks=[1, 2, 4, 6, 2],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DarkNetC3_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv5 version medium
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
    """
    
    model = DarkNetC3_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

        
def DarkNetC3_large(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C3,
        pyramid_pooling=SPP,
        filters=64,
        num_blocks=[1, 3, 6, 9, 3],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DarkNetC3_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv5 version large
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5l.yaml
    """
    
    model = DarkNetC3_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

        
def DarkNetC3_xlarge(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetC3(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C3,
        pyramid_pooling=SPP,
        filters=80,
        num_blocks=[1, 4, 8, 12, 4],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate
    )
    return model


def DarkNetC3_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv5 version xlarge
        - In YOLOv5, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/yolov5/blob/master/models/yolov5x.yaml
    """

    model = DarkNetC3_xlarge(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
