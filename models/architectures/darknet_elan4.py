"""
  # Description:
    - The following table comparing the params of the DarkNet ELAN4 (YOLOv9 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       --------------------------------------------
      |      Model Name          |    Params       |
      |--------------------------------------------|
      |    DarkNetELAN small     |    8,350,400    |
      |--------------------------------------------|
      |    DarkNetELAN base      |   12,184,256    |
      |--------------------------------------------|
      |    DarkNetELAN large     |   56,867,240    |
       --------------------------------------------

  # Reference:
    - Source: https://github.com/WongKinYiu/yolov9

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
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add

from .darknet53 import ConvolutionBlock
from .darknet_c2 import Bottleneck2
from .darknet_c3 import Bottleneck, BottleneckCSP
from .darknet_elan import BottleneckCSPA

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


class AverageConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters, 
                 kernel_size       = 3, 
                 activation        = 'leaky', 
                 norm_layer        = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(AverageConvolutionBlock, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.activation    = activation
        self.norm_layer    = norm_layer

    def build(self, input_shape):
        self.conv = ConvolutionBlock(filters=self.filters,
                                     kernel_size=self.kernel_size,
                                     downsample=True, 
                                     activation=self.activation,
                                     norm_layer=self.norm_layer)
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="SAME")
        
    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv(x, training=training)
        return x


class AverageConvolutionDown(tf.keras.layers.Layer):

    def __init__(self,
                 filters, 
                 kernel_size       = 3, 
                 activation        = 'leaky', 
                 norm_layer        = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(AverageConvolutionDown, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.activation    = activation
        self.norm_layer    = norm_layer

    def build(self, input_shape):
        hidden_dim = self.filters // 2
        self.conv1 = ConvolutionBlock(filters=hidden_dim,
                                     kernel_size=self.kernel_size,
                                     downsample=True, 
                                     activation=self.activation,
                                     norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(filters=hidden_dim,
                                      kernel_size=(1, 1),
                                      activation=self.activation,
                                      norm_layer=self.norm_layer)
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="SAME")
        self.max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")
        
    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        x1 = self.conv1(x1, training=training)
        x2 = self.max_pool(x2, training=training)
        x2 = self.conv2(x2, training=training)
        x = concatenate([x1, x2], axis=-1)
        return x


class RepSimpleBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 downsample=False,
                 dilation_rate=1,
                 groups=1,
                 activation='silu', 
                 norm_layer='batch-norm',
                 *args, 
                 **kwargs):
        super(RepSimpleBlock, self).__init__(*args, **kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.downsample    = downsample
        self.dilation_rate = dilation_rate
        self.groups        = groups
        self.activation    = activation
        self.norm_layer    = norm_layer

    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      downsample=self.downsample, 
                                      dilation_rate=self.dilation_rate,
                                      groups=self.groups,
                                      activation=None,
                                      norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(filters=self.filters,
                                      kernel_size=(1, 1),
                                      downsample=self.downsample, 
                                      groups=self.groups,
                                      activation=None,
                                      norm_layer=self.norm_layer)
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)        
        x = x1 + x2
        x = self.activ(x, training=training)
        return x


class RepBottleneck(Bottleneck):
    
    def __init__(self,
                 filters, 
                 kernels    = (3, 3),
                 downsample = False,
                 groups     = 1,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         downsample,
                         groups,
                         expansion,
                         shortcut,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)
        self.kernels = kernels
        
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = RepSimpleBlock(hidden_dim, self.kernels[0], activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.filters, self.kernels[1], downsample=self.downsample, groups=self.groups, activation=self.activation, norm_layer=self.norm_layer)


class ResNBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 filters, 
                 downsample = False,
                 groups     = 1,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(ResNBlock, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.downsample = downsample
        self.groups     = groups
        self.expansion  = expansion
        self.shortcut   = shortcut
        self.activation = activation
        self.norm_layer = norm_layer

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(hidden_dim, (3, 3), activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.filters, (1, 1), downsample=self.downsample, groups=self.groups, activation=self.activation, norm_layer=self.norm_layer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class RepResNBlock(ResNBlock):
    
    def __init__(self,
                 filters, 
                 downsample = False,
                 groups     = 1,
                 expansion  = 1,
                 shortcut   = True,
                 activation = 'silu', 
                 norm_layer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super().__init__(filters, 
                         downsample,
                         groups,
                         expansion,
                         shortcut,
                         activation,
                         norm_layer,
                         *args,
                         **kwargs)
        
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = RepSimpleBlock(hidden_dim, (3, 3), activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.filters, (1, 1), downsample=self.downsample, groups=self.groups, activation=self.activation, norm_layer=self.norm_layer)


class BottleneckCSPA2(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            Bottleneck2(hidden_dim, groups=self.groups, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class BottleneckCSP2(BottleneckCSP):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.middle = Sequential([
            Bottleneck2(hidden_dim, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class RepNCSP(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            RepBottleneck(hidden_dim, kernels=(3, 3), groups=self.groups, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class BaseBottleneckCSP(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.block = Sequential([
            Bottleneck(hidden_dim, groups=self.groups, shortcut=self.shortcut, activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])


class ASPP(tf.keras.layers.Layer):
    
    def __init__(self, 
                 filters, 
                 **kwargs):
        super(ASPP, self).__init__(**kwargs)
        self.filters       = filters
        self.kernel_list   = [1, 3, 3, 1]
        self.dilation_list = [1, 3, 6, 1]
        self.padding_list  = [0, 1, 1, 0]

    def build(self, input_shape):
        self.gap   = GlobalAveragePooling2D(keepdims=True)
        self.block = [Conv2D(filters=self.filters, 
                             kernel_size=k, 
                             strides=1, 
                             padding="SAME" if p else "VALID", 
                             dilation_rate=d, 
                             use_bias=True) for k, p, d in zip(self.kernel_list, self.padding_list, self.dilation_list)]

    def call(self, inputs, training=False):
        avg_x = self.gap(inputs)

        out = []
        for idx, bk in enumerate(self.block):
            y = inputs if idx != len(self.block) - 1 else avg_x
            y = bk(y, training=training)
            y = tf.nn.relu(y)
            out.append(y)

        out[-1] = tf.broadcast_to(out[-1], tf.shape(out[-2]))
        out = concatenate(out, axis=-1)
        return out


class SPPELAN(tf.keras.layers.Layer):

    def __init__(self, 
                 filters, 
                 activation='relu',
                 norm_layer='batch-norm',
                 **kwargs):
        super(SPPELAN, self).__init__(**kwargs)
        self.filters    = filters
        self.activation = activation
        self.norm_layer = norm_layer
                    
    def build(self, input_shape):
        if isinstance(self.filters, (tuple, list)):
            f0, f1 = self.filters
        else:
            f0 = f1 = self.filters
        self.conv1 = ConvolutionBlock(f0, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        self.block = [MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME"),
                      MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME"),
                      MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME")]
        self.conv2 = ConvolutionBlock(f1, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        
        y = self.conv1(inputs, training=training)
        out = [y]
        for bk in self.block:
            out.append(bk(out[-1], training=training))

        x = concatenate(out, axis=-1)
        x = self.conv2(x, training=training)
        return x


class RepNCSPELAN4(tf.keras.layers.Layer):

    def __init__(self, 
                 filters, 
                 iters      = 1,
                 groups     = 1,
                 expansion  = 0.5,
                 shortcut   = True,
                 activation = 'relu',
                 norm_layer = 'batch-norm',
                 **kwargs):
        super(RepNCSPELAN4, self).__init__(**kwargs)
        self.filters    = filters
        self.iters      = iters
        self.groups     = groups
        self.expansion  = expansion
        self.shortcut   = shortcut
        self.activation = activation
        self.norm_layer = norm_layer
                    
    def build(self, input_shape):
        if isinstance(self.filters, (tuple, list)):
            f0, f1, f2 = self.filters
        else:
            f0 = f1, f2 = self.filters
        self.conv1 = ConvolutionBlock(f0, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        self.block = [
            Sequential([
                RepNCSP(f1, 
                        groups     = self.groups, 
                        iters      = self.iters,
                        expansion  = self.expansion,
                        shortcut   = self.shortcut,
                        activation = self.activation, 
                        norm_layer = self.norm_layer),
                ConvolutionBlock(f1, (3, 3), activation=self.activation, norm_layer=self.norm_layer)
            ]),
            Sequential([
                RepNCSP(f1, 
                        groups     = self.groups, 
                        iters      = self.iters,
                        expansion  = self.expansion,
                        shortcut   = self.shortcut,
                        activation = self.activation, 
                        norm_layer = self.norm_layer),
                ConvolutionBlock(f1, (3, 3), activation=self.activation, norm_layer=self.norm_layer)
            ])
        ]
        self.conv2 = ConvolutionBlock(f2, (1, 1), activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        out = self.conv1(inputs, training=training)
        out = tf.split(out, num_or_size_splits=2, axis=-1)
        
        for bk in self.block:
            out.append(bk(out[-1], training=training))

        x = concatenate(out, axis=-1)
        x = self.conv2(x, training=training)
        return x


class CBLinear(tf.keras.layers.Layer):

    def __init__(self, 
                 split_filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding="SAME",
                 groups=1,
                 activation='relu',
                 norm_layer='batch-norm',
                 **kwargs):
        super(CBLinear, self).__init__(**kwargs)
        self.split_filters = split_filters
        self.kernel_size   = kernel_size
        self.strides       = strides
        self.padding       = padding
        self.groups        = groups
        self.activation    = activation
        self.norm_layer    = norm_layer
                    
    def build(self, input_shape):
        self.conv1 = Conv2D(filters=sum(self.split_filters),
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding=self.padding,
                            groups=self.groups,
                            use_bias=True)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = tf.split(x, num_or_size_splits=self.split_filters, axis=-1)
        return x


class CBFuse(tf.keras.layers.Layer):

    def __init__(self, 
                 fuse_index,
                 **kwargs):
        super(CBFuse, self).__init__(**kwargs)
        self.fuse_index = fuse_index
                    
    def build(self, input_shape):
        self.target_size = input_shape[0][1:-1]

    def call(self, inputs, training=False):
        res = []
        for idx, feature in enumerate(inputs[1:]):
            x = tf.image.resize(feature[self.fuse_index[idx]], size=self.target_size, method=tf.image.ResizeMethod.BILINEAR)
            res.append(x)
            
        res.append(inputs[0])
        out = tf.stack(res, axis=0)
        out = tf.reduce_sum(out, axis=0)
        return out

        
def DarkNetELAN4_A(down_block,
                   filters=[64, 128, 256, 512],
                   num_blocks=[1, 1, 1, 1], 
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
        
    f0, f1, f2, f3 = filters
    n0, n1, n2, n3 = num_blocks

    # conv down
    x = ConvolutionBlock(f0, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer, name='stem.block1')(img_input)
                       
    # conv down
    x = ConvolutionBlock(f1, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer, name='stage1.block1')(x)
                       
    # elan-1 block
    x = RepNCSPELAN4(filters    = [f1, f0, f2], 
                     iters      = n0,
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer,
                     name       = 'stage1.block2')(x)

    # conv down
    if down_block == AverageConvolutionDown:
        x = down_block(f2, (3, 3), activation=activation, norm_layer=norm_layer, name='stage2.block1')(x)
    elif down_block == ConvolutionBlock:
        x = down_block(f2, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer, name='stage2.block1')(x)

    # elan-2 block
    x = RepNCSPELAN4(filters    = [f2, f1, f3], 
                     iters      = n1,
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer,
                     name       = 'stage2.block2')(x)

    # conv down
    if down_block == AverageConvolutionDown:
        x = down_block(f3, (3, 3), activation=activation, norm_layer=norm_layer, name='stage3.block1')(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f3, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer, name='stage3.block1')(x)

    # elan-3 block
    x = RepNCSPELAN4(filters    = [f3, f2, f3], 
                     iters      = n2,
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer,
                     name       = 'stage3.block2')(x)

    # conv down
    if down_block == AverageConvolutionDown:
        x = down_block(f3, (3, 3), activation=activation, norm_layer=norm_layer, name='stage4.block1')(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f3, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer, name='stage4.block1')(x)

    # elan-4 block
    x = RepNCSPELAN4(filters    = [f3, f2, f3], 
                     iters      = n3,
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer,
                     name       = 'stage4.block2')(x)

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
    if down_block == AverageConvolutionDown and filters == [64, 128, 256, 512] and num_blocks == [1, 1, 1, 1]:
        model = Model(inputs, x, name='DarkNet-ELAN4-small')
    elif down_block == ConvolutionBlock and filters == [64, 128, 256, 512] and num_blocks == [1, 1, 1, 1]:
        model = Model(inputs, x, name='DarkNet-ELAN4-base')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN4-A')

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


def DarkNetELAN4_B(down_block,
                   filters=[64, 128, 256, 512, 1024],
                   num_blocks=[2, 2, 2, 2, 2, 2, 2, 2], 
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
        
    f0, f1, f2, f3, f4 = filters

    y0 = ConvolutionBlock(f0, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(img_input)

    x = ConvolutionBlock(f0, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(img_input)
    y1 = CBLinear([f0])(x)

    x = ConvolutionBlock(f1, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)
                       
    x = RepNCSPELAN4(filters    = [f1, f0, f2], 
                     iters      = num_blocks[0],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)
    y2 = CBLinear([f0, f1])(x)
        
    if down_block == AverageConvolutionDown:
        x = down_block(f2, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = down_block(f2, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = RepNCSPELAN4(filters    = [f2, f1, f3], 
                     iters      = num_blocks[1],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)
    y3 = CBLinear([f0, f1, f2])(x)
        
    if down_block == AverageConvolutionDown:
        x = down_block(f3, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f3, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = RepNCSPELAN4(filters    = [f3, f2, f4], 
                     iters      = num_blocks[2],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)
    y4 = CBLinear([f0, f1, f2, f3])(x)

    if down_block == AverageConvolutionDown:
        x = down_block(f4, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f4, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = RepNCSPELAN4(filters    = [f3, f2, f4], 
                     iters      = num_blocks[3],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)
    y5 = CBLinear([f0, f1, f2, f3, f4])(x)

    x = CBFuse(fuse_index=[0, 0, 0, 0, 0])([y0, y1, y2, y3, y4, y5])

    x = ConvolutionBlock(f1, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)
    x = CBFuse(fuse_index=[1, 1, 1, 1, 1])([x, y2, y3, y4, y5])

    x = RepNCSPELAN4(filters    = [f1, f0, f2], 
                     iters      = num_blocks[4],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)

    if down_block == AverageConvolutionDown:
        x = down_block(f2, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f2, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = CBFuse(fuse_index=[2, 2, 2])([x, y3, y4, y5])

    x = RepNCSPELAN4(filters    = [f2, f1, f3], 
                     iters      = num_blocks[5],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)

    if down_block == AverageConvolutionDown:
        x = down_block(f3, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f3, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = CBFuse(fuse_index=[3, 3])([x, y4, y5])

    x = RepNCSPELAN4(filters    = [f3, f2, f4], 
                     iters      = num_blocks[6],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)

    if down_block == AverageConvolutionDown:
        x = down_block(f4, (3, 3), activation=activation, norm_layer=norm_layer)(x)
    elif down_block == ConvolutionBlock:
        x = ConvolutionBlock(f4, (3, 3), downsample=True, activation=activation, norm_layer=norm_layer)(x)

    x = CBFuse(fuse_index=[4])([x, y5])

    x = RepNCSPELAN4(filters    = [f3, f2, f4], 
                     iters      = num_blocks[7],
                     groups     = 1,
                     expansion  = 0.5,
                     shortcut   = True,
                     activation = activation,
                     norm_layer = norm_layer)(x)

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
    if down_block == AverageConvolutionDown and filters == [64, 128, 256, 512, 1024] and num_blocks == [2, 2, 2, 2, 2, 2, 2, 2]:
        model = Model(inputs, x, name='DarkNet-ELAN4-Large')
    else:
        model = Model(inputs, x, name='DarkNet-ELAN4-B')

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


def DarkNetELAN4_small(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       norm_layer='batch-norm',
                       final_activation="softmax",
                       classes=1000) -> Model:
    
    model = DarkNetELAN4_A(down_block=AverageConvolutionDown,
                           filters=[64, 128, 256, 512],
                           num_blocks=[1, 1, 1, 1], 
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


def DarkNetELAN4_small_backbone(input_shape=(640, 640, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='leaky-relu',
                                norm_layer='batch-norm',
                                custom_layers=None) -> Model:

    """
        - Used in YOLOv9-C
        - In YOLOv9, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov9/blob/main/models/detect/yolov9-c.yaml
    """
    
    model = DarkNetELAN4_small(include_top=include_top, 
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
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')

        
def DarkNetELAN4_base(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      norm_layer='batch-norm',
                      final_activation="softmax",
                      classes=1000) -> Model:
    
    model = DarkNetELAN4_A(down_block=ConvolutionBlock,
                           filters=[64, 128, 256, 512],
                           num_blocks=[1, 1, 1, 1], 
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


def DarkNetELAN4_base_backbone(input_shape=(640, 640, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='leaky-relu',
                               norm_layer='batch-norm',
                               custom_layers=None) -> Model:

    """
        - Used in YOLOv9
        - In YOLOv9, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov9/blob/main/models/detect/yolov9.yaml
    """
    
    model = DarkNetELAN4_base(include_top=include_top, 
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
        y_4 = model.get_layer("stage1.block2").output
        y_8 = model.get_layer("stage2.block2").output
        y_16 = model.get_layer("stage3.block2").output
        y_32 = model.get_layer("stage4.block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def DarkNetELAN4_Large(include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       norm_layer='batch-norm',
                       final_activation="softmax",
                       classes=1000) -> Model:
    
    model = DarkNetELAN4_B(down_block=AverageConvolutionDown,
                           filters=[64, 128, 256, 512, 1024],
                           num_blocks=[2, 2, 2, 2, 2, 2, 2, 2], 
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