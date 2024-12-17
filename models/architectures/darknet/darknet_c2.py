"""
  # Description:
    - The following table comparing the params of the DarkNet 53 with C2f Block (YOLOv8 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       ----------------------------------------
      |      Model Name      |    Params       |
      |----------------------------------------|
      |    DarkNetC2 nano    |    1,534,680    |
      |----------------------------------------|
      |    DarkNetC2 small   |    5,602,760    |
      |----------------------------------------|
      |    DarkNetC2 medium  |   12,449,464    |
      |----------------------------------------|
      |    DarkNetC2 large   |   20,344,744    |
      |----------------------------------------|
      |    DarkNetC2 xlarge  |   31,613,080    |
       ----------------------------------------

  # Reference:
    - Source: https://github.com/ultralytics/ultralytics

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
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from .darknet53 import ConvolutionBlock
from .darknet_c3 import Bottleneck, C3, SPP, SPPF
from models.layers import get_activation_from_name, get_normalizer_from_name, RepVGGBlock
from utils.model_processing import _obtain_input_shape




class SimpleRepVGG(tf.keras.layers.Layer):
    
    '''
        Simplified RepConv module with Conv fusing
    '''
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="same",
                 dilation_rate=(1, 1),
                 groups=1,
                 activation='relu', 
                 normalizer='batch-norm',
                 training=False,
                 *args, 
                 **kwargs):
        super(SimpleRepVGG, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides       = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding       = padding
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (tuple, list)) else (dilation_rate, dilation_rate)
        self.groups        = groups
        self.activation    = activation
        self.normalizer    = normalizer
        self.training      = training

    def build(self, input_shape):
        self.conv1 = Conv2D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate,
                            groups=self.groups,
                            use_bias=False)
        
        self.conv2 = Conv2D(filters=self.filters,
                            kernel_size=1,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate,
                            groups=self.groups,
                            use_bias=False)
        self.norm  = get_normalizer_from_name(self.normalizer)
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        out = x1 + x2
        out = self.norm(out)
        out = self.activ(out)
        return out


class LightConvolutionBlock(tf.keras.layers.Layer):
    
    '''
        Light convolution. https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    '''
    
    def __init__(self,
                 filters,
                 kernel_size,
                 activation='relu', 
                 normalizer='batch-norm',
                 *args, 
                 **kwargs):
        super(LightConvolutionBlock, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.activation    = activation
        self.normalizer    = normalizer

    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, self.kernel_size, groups=self.filters, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x


class ChannelAttention(tf.keras.layers.Layer):
    
    '''
        Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    '''
    
    def __init__(self, activation='sigmoid', *args, **kwargs):
        super(ChannelAttention, self).__init__(*args, **kwargs)
        self.activation = activation
        
    def build(self, input_shape):
        self.pool = GlobalAveragePooling2D(keepdims=True)
        self.conv = ConvolutionBlock(input_shape[-1], 1, activation=self.activation, normalizer=None)

    def call(self, inputs, training=False):
        x = self.pool(inputs)
        x = self.conv(x, training=training)
        return inputs * x


class SpatialAttention(tf.keras.layers.Layer):
    
    '''
        Spatial-attention module.
    '''
    
    def __init__(self, kernel_size, activation='sigmoid', *args, **kwargs):
        super(SpatialAttention, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.activation  = activation

    def build(self, input_shape):
        self.conv = ConvolutionBlock(1, self.kernel_size, activation=self.activation, normalizer=None)
        
    def call(self, inputs, training=False):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        merger = concatenate([mean, max], axis=-1)
        x = self.conv(merger, training=training)
        return inputs * x


class CBAM(tf.keras.layers.Layer):
    
    '''
        Convolutional Block Attention Module.
    '''
    
    def __init__(self, kernel_size, activation='sigmoid', *args, **kwargs):
        super(CBAM, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.activation  = activation

    def build(self, input_shape):
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention(self.kernel_size)

    def call(self, inputs, training=False):
        x = self.channel_attention(inputs, training=training)
        x = self.spatial_attention(x, training=training)
        return x


class Conv2DCustomKernel(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size=(1, 1), strides=(1, 1), padding="VALID", dilations=1, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding
        self.dilations = dilations
        super(Conv2DCustomKernel, self).__init__(**kwargs)
        
    def build(self, input_shape):
        kernel_init = tf.range(0, self.filters, 1, dtype=tf.float32)
        shape = list(self.kernel_size) + [1, self.filters]
        kernel_init = tf.reshape(kernel_init, shape=shape)
        self.kernel = tf.Variable(name="kernel_value",
                                   initial_value=kernel_init,
                                   dtype=tf.float32,
                                   trainable=False)
        super(Conv2DCustomKernel, self).build(input_shape)
        
    def call(self, inputs, training=False):
        x = tf.stop_gradient(tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, dilations=self.dilations))
        return x


class DFL(tf.keras.layers.Layer):
    
    '''
        Integral module of Distribution Focal Loss (DFL).
        Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    '''
    
    def __init__(self, filters, *args, **kwargs):
        super(DFL, self).__init__(*args, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        bs = input_shape[-1]
        self.conv = Conv2DCustomKernel(filters=self.filters,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding='VALID')

    def call(self, inputs, training=False):        
        _, anchor, dim = inputs.shape
        x = tf.reshape(inputs, shape=(-1, anchor, 4, self.filters))
        x = tf.nn.softmax(x, axis=-1)
        x = self.conv(x, training=training)
        x = tf.reshape(x, shape=(-1, 4, anchor))
        return x



class Proto(tf.keras.layers.Layer):
    
    '''
        YOLOv8 mask Proto module for segmentation models.
    '''
    
    def __init__(self, filters, expansion=8, activation='silu', normalizer='batch-norm', *args, **kwargs):
        super(Proto, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 3, activation=self.activation, normalizer=self.normalizer)
        self.upsample = Conv2DTranspose(filters=hidden_dim,
                                        kernel_size=(2, 2),
                                        strides=(2, 2),
                                        padding="same",
                                        use_bias=True)
        self.conv2 = ConvolutionBlock(hidden_dim, 3, activation=self.activation, normalizer=self.normalizer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)


    def call(self, inputs, training=False):        
        x = self.conv1(inputs, training=training)
        x = self.upsample(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class HGStem(tf.keras.layers.Layer):
    
    '''
        StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    '''
    
    def __init__(self, filters, expansion=8, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(HGStem, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = self.convolution_block(hidden_dim, 3, strides=(2, 2), padding="same")
        self.conv2 = self.convolution_block(hidden_dim // 2, 2, strides=(1, 1))
        self.conv3 = self.convolution_block(hidden_dim, 2, strides=(1, 1))
        self.pool  = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")
        self.conv4 = self.convolution_block(hidden_dim, 3, strides=(2, 2))
        self.conv5 = self.convolution_block(self.filters, 1, strides=(1, 1))
        
    def convolution_block(self, filters, kernel_size, strides=1, padding="valid", groups=1):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       groups=groups,
                       use_bias=False),
                get_normalizer_from_name(self.normalizer),
                get_activation_from_name(self.activation)
        ])
        
    def call(self, inputs, training=False):        
        x = self.conv1(inputs, training=training)
        pad = tf.constant([[0, 0,], [0, 1], [0, 1], [0, 0]])
        x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)
        x1 = self.conv2(x, training=training)
        x1 = tf.pad(x1, pad, mode='CONSTANT', constant_values=0)
        x1 = self.conv3(x1, training=training)
        x2 = self.pool(x)
        x = concatenate([x1, x2], axis=-1)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        return x



class HGBlock(tf.keras.layers.Layer):
    
    '''
        HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    '''
    
    def __init__(self, filters, kernel_size=3, iters=6, expansion=2, shortcut=False, block=ConvolutionBlock, activation='relu', normalizer='batch-norm', *args, **kwargs):
        super(HGBlock, self).__init__(*args, **kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.iters       = iters
        self.expansion   = expansion
        self.shortcut    = shortcut
        self.block       = block
        self.activation  = activation
        self.normalizer  = normalizer

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.module_list = [self.block(hidden_dim, self.kernel_size, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)]
        self.sc = ConvolutionBlock(self.filters // 2, 1, activation=self.activation, normalizer=self.normalizer)
        self.ec = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):        
        x = [inputs]
        for module in self.module_list:
            x.append(module(x[-1]))

        x = concatenate(x, axis=-1)
        x = self.sc(x, training=training)
        x = self.ec(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class C1(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck with 1 convolution. 
    """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(C1, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.middle = Sequential([
            ConvolutionBlock(self.filters, 3, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        y = self.middle(x, training=training)
        if self.shortcut:
            y = add([x, y])
        return y


class Bottleneck2(Bottleneck):
    
    def __init__(self,
                 filters, 
                 kernels    = (3, 3),
                 downsample = False,
                 groups     = 1,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         downsample,
                         groups,
                         expansion,
                         shortcut,
                         activation,
                         normalizer,
                         *args,
                         **kwargs)
        self.kernels = kernels
        
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, self.kernels[0], activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, self.kernels[1], downsample=self.downsample, groups=self.groups, activation=self.activation, normalizer=self.normalizer)



class C2(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck with 2 convolution. 
    """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(C2, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(2 * hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        
        self.blocks = Sequential([
            Bottleneck2(hidden_dim, (3, 3), shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        x1 = self.blocks(x1, training=training)
        x = concatenate([x1, x2], axis=-1)
        x = self.conv2(x, training=training)
        return x


class C2f(tf.keras.layers.Layer):
    
    """ 
        Faster Implementation of CSP Bottleneck with 2 convolutions. 
    """
    
    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 **kwargs):
        super(C2f, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.expansion  = expansion
        self.shortcut   = shortcut       
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(2 * hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        
        # self.blocks = Sequential([
        #     Bottleneck2(hidden_dim, (3, 3), shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)
        # ])
        self.blocks = [Bottleneck2(hidden_dim, (3, 3), shortcut=self.shortcut, activation=self.activation, normalizer=self.normalizer) for i in range(self.iters)]

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = tf.split(x, num_or_size_splits=2, axis=-1)
        for block in self.blocks:
            x.append(block(x[-1]))
        x = concatenate(x, axis=-1)
        x = self.conv2(x, training=training)
        return x


class C3Rep(C3):
    
    """ 
        C3 module with Rep-convolutions. 
    """

    def __init__(self, 
                 filters, 
                 iters,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 normalizer = 'batch-norm', 
                 training   = False,
                 *args,
                 **kwargs):
        super().__init__(filters, 
                         iters,
                         expansion,
                         shortcut,
                         activation,
                         normalizer,
                         *args,
                         **kwargs)
        self.training = training
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.middle = Sequential([
            RepVGGBlock(hidden_dim, kernel_size=3, training=self.training) for i in range(self.iters)
        ])
        self.residual = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        if hidden_dim != self.filters:
            self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.middle(x, training=training)
        y = self.residual(inputs, training=training)
        
        merger = add([x, y])
        if self.shortcut and hasattr(self, 'conv2'):
            merger = self.conv2(merger, training=training)
        return merger


def DarkNetC2(c2_block,
              spp_block,
              layers,
              filters,
              scale_ratio=1,
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
            
    x = ConvolutionBlock(filters, 3, downsample=True, activation=activation, normalizer=normalizer, name='stem')(img_input)
    
    x = ConvolutionBlock(filters * 2, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage1.block1')(x)
    x = c2_block(filters * 2, l0, activation=activation, normalizer=normalizer, name='stage1.block2')(x)

    x = ConvolutionBlock(filters * 4, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage2.block1')(x)
    x = c2_block(filters * 4, l1, activation=activation, normalizer=normalizer, name='stage2.block2')(x)

    x = ConvolutionBlock(filters * 8, 3, downsample=True, activation=activation, normalizer=normalizer, name='stage3.block1')(x)
    x = c2_block(filters * 8, l2, activation=activation, normalizer=normalizer, name='stage3.block2')(x)

    x = ConvolutionBlock(int(filters * 16 * scale_ratio), 3, downsample=True, activation=activation, normalizer=normalizer, name='stage4.block1')(x)
    x = c2_block(int(filters * 16 * scale_ratio), l3, activation=activation, normalizer=normalizer, name='stage4.block2')(x)
    x = spp_block(int(filters * 16 * scale_ratio), name='stage4.block3')(x)

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
    if layers == [1, 2, 2, 1] and filters == 16:
        model = Model(inputs, x, name='DarkNet-C2-Nano')
    elif layers == [1, 2, 2, 1] and filters == 32:
        model = Model(inputs, x, name='DarkNet-C2-Small')
    elif layers == [2, 4, 4, 2] and filters == 48:
        model = Model(inputs, x, name='DarkNet-C2-Medium')
    elif layers == [3, 6, 6, 3] and filters == 64:
        model = Model(inputs, x, name='DarkNet-C2-Large')
    elif layers == [3, 6, 6, 3] and filters == 80:
        model = Model(inputs, x, name='DarkNet-C2-XLarge')
    else:
        model = Model(inputs, x, name='DarkNet-C2')

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


def DarkNetC2_nano(c2_block=C2f,
                   spp_block=SPPF,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   activation='silu',
                   normalizer='batch-norm',
                   final_activation="softmax",
                   classes=1000) -> Model:
    
    model = DarkNetC2(c2_block=c2_block,
                      spp_block=spp_block,
                      layers=[1, 2, 2, 1],
                      filters=16,
                      scale_ratio=1,
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


def DarkNetC2_nano_backbone(c2_block=C2f,
                            spp_block=SPPF,
                            input_shape=(640, 640, 3),
                            include_top=False, 
                            weights='imagenet', 
                            activation='silu',
                            normalizer='batch-norm',
                            custom_layers=None) -> Model:
    
    """
        - Used in YOLOv8 version nano
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """
    
    model = DarkNetC2_nano(c2_block=c2_block,
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


def DarkNetC2_small(c2_block=C2f,
                    spp_block=SPPF,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:
    
    model = DarkNetC2(c2_block=c2_block,
                      spp_block=spp_block,
                      layers=[1, 2, 2, 1],
                      filters=32,
                      scale_ratio=1,
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


def DarkNetC2_small_backbone(c2_block=C2f,
                             spp_block=SPPF,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:
    
    """
        - Used in YOLOv8 version small
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """
    
    model = DarkNetC2_small(c2_block=c2_block,
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


def DarkNetC2_medium(c2_block=C2f,
                     spp_block=SPPF,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetC2(c2_block=c2_block,
                      spp_block=spp_block,
                      layers=[2, 4, 4, 2],
                      filters=48,
                      scale_ratio=0.75,
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


def DarkNetC2_medium_backbone(c2_block=C2f,
                              spp_block=SPPF,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:
    
    """
        - Used in YOLOv8 version medium
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """
    
    model = DarkNetC2_medium(c2_block=c2_block,
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


def DarkNetC2_large(c2_block=C2f,
                    spp_block=SPPF,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    activation='silu',
                    normalizer='batch-norm',
                    final_activation="softmax",
                    classes=1000) -> Model:
    
    model = DarkNetC2(c2_block=c2_block,
                      spp_block=spp_block,
                      layers=[3, 6, 6, 3],
                      filters=64,
                      scale_ratio=0.5,
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


def DarkNetC2_large_backbone(c2_block=C2f,
                             spp_block=SPPF,
                             input_shape=(640, 640, 3),
                             include_top=False, 
                             weights='imagenet', 
                             activation='silu',
                             normalizer='batch-norm',
                             custom_layers=None) -> Model:
    
    """
        - Used in YOLOv8 version large
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """
    
    model = DarkNetC2_large(c2_block=c2_block,
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


def DarkNetC2_xlarge(c2_block=C2f,
                     spp_block=SPPF,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     activation='silu',
                     normalizer='batch-norm',
                     final_activation="softmax",
                     classes=1000) -> Model:
    
    model = DarkNetC2(c2_block=c2_block,
                      spp_block=spp_block,
                      layers=[3, 6, 6, 3],
                      filters=80,
                      scale_ratio=0.5,
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


def DarkNetC2_xlarge_backbone(c2_block=C2f,
                              spp_block=SPPF,
                              input_shape=(640, 640, 3),
                              include_top=False, 
                              weights='imagenet', 
                              activation='silu',
                              normalizer='batch-norm',
                              custom_layers=None) -> Model:
    
    """
        - Used in YOLOv8 version xlarge
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """
    
    model = DarkNetC2_xlarge(c2_block=c2_block,
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
