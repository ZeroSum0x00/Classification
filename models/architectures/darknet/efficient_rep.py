"""
  # Description:
    - The following table comparing the params of the EfficientRep architectures (YOLOv6 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       ---------------------------------------------------
      |         Model Name             |     Params       |
      |---------------------------------------------------|
      |    Efficient-Rep lite nano     |       345,070    |
      |---------------------------------------------------|
      |    Efficient-Rep lite medium   |       676,605    |
      |---------------------------------------------------|
      |    Efficient-Rep lite large    |     1,066,285    |
      |---------------------------------------------------|
      |    Efficient-Rep nano          |     3,393,480    |
      |---------------------------------------------------|
      |    Efficient-Rep6 nano         |     4,426,248    |
      |---------------------------------------------------|
      |    Efficient-Rep small         |    13,045,672    |
      |---------------------------------------------------|
      |    Efficient-Rep6 small        |    17,175,592    |
      |---------------------------------------------------|
      |    Efficient-Rep medium        |    24,251,732    |
      |---------------------------------------------------|
      |    Efficient-Rep6 medium       |    33,770,134    |
      |---------------------------------------------------|
      |    Efficient-Rep large         |    39,531,453    |
      |---------------------------------------------------|
      |    Efficient-Rep6 large        |    54,590,400    |
      |---------------------------------------------------|
      |    Efficient-MBLA small        |     7,252,048    |
      |---------------------------------------------------|
      |    Efficient-MBLA medium       |    15,911,808    |
      |---------------------------------------------------|
      |    Efficient-MBLA large        |    27,927,728    |
      |---------------------------------------------------|
      |    Efficient-MBLA xlarge       |    51,591,610    |
       ----------------------------------------------------

  # Reference:
    - Source: https://github.com/meituan/YOLOv6/tree/main

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
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file

from .darknet53 import ConvolutionBlock
from .darknet_c3 import SPPF
from models.layers import get_activation_from_name, get_normalizer_from_name, ChannelShuffle, RepVGGBlock, ScaleWeight
from utils.model_processing import _obtain_input_shape


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self,
                 expansion    = 0.5,
                 rep_block    = RepVGGBlock,
                 sub_block    = None,
                 scale_weight = False,
                 iters        = 1,
                 activation   = 'relu', 
                 normalizer   = 'batch-norm', 
                 training     = False,
                 *args, 
                 **kwargs):
        super(CustomLayer, self).__init__(*args, **kwargs)
        self.expansion    = expansion
        self.rep_block    = rep_block
        self.sub_block    = sub_block 
        self.scale_weight = scale_weight
        self.iters        = iters
        self.activation   = activation
        self.normalizer   = normalizer
        self.training     = training


class CSPSPPF(CustomLayer):
    
    """ 
        CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    
    def __init__(self, 
                 filters, 
                 pool_size  = (5, 5),
                 expansion  = 0.5,
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(CSPSPPF, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.pool_size  = pool_size
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
                     
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(hidden_dim, 3, activation=self.activation, normalizer=self.normalizer)
        self.conv3 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding='same')
        self.conv4 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv5 = ConvolutionBlock(hidden_dim, 3, activation=self.activation, normalizer=self.normalizer)
        self.conv6 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)

        self.shortcut = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        x = concatenate([x, p1, p2, p3], axis=-1)

        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        y = self.shortcut(inputs, training=training)

        out = concatenate([y, x], axis=-1)
        out = self.conv6(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "pool_size": self.pool_size,
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer
        })
        return config


class LinearAddBlock(CustomLayer):
    
    '''
        A CSLA block is a LinearAddBlock with is_csla=True
    '''
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding="same",
                 dilation=1,
                 groups=1,
                 is_csla=False,
                 conv_scale_init=1.0,
                 activation='relu', 
                 normalizer='batch-norm',
                 *args, 
                 **kwargs):
        super(LinearAddBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.is_csla = is_csla
        self.conv_scale_init = conv_scale_init
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        self.activ = get_activation_from_name(self.activation)
        self.norm = get_normalizer_from_name(self.normalizer)
        self.conv  = Conv2D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding=self.padding,
                            use_bias=False)
        self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
        self.conv_1x1 = Conv2D(filters=self.filters,
                            kernel_size=(1, 1),
                            strides=self.strides,
                            padding='valid',
                            use_bias=False)
        self.scale_1x1 = ScaleWeight(self.conv_scale_init, use_bias=False)
        
        if input_shape[-1] == self.filters and self.strides == (1, 1):
            self.scale_identity = ScaleWeight(1.0, use_bias=False)

        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.scale_layer(x, training=False if self.is_csla else training)
        y = self.conv_1x1(inputs, training=training)
        y = self.scale_1x1(y, training=False if self.is_csla else training)
        out = x + y
        if hasattr(self, 'scale_identity'):
            out += self.scale_identity(inputs, training=training)
        out = self.norm(out, training=training)
        out = self.activ(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "is_csla": self.is_csla,
            "conv_scale_init": self.conv_scale_init,
            "activation": self.activation,
            "normalizer": self.normalizer
        })
        return config


class BottleRep(CustomLayer):
    
    def __init__(self, filters, rep_block=RepVGGBlock, scale_weight=False, training=False, *args, **kwargs):
        super(BottleRep, self).__init__(*args, **kwargs)
        self.filters      = filters
        self.rep_block    = rep_block
        self.scale_weight = scale_weight
        self.training     = training

    def build(self, input_shape):
        if self.rep_block == ConvolutionBlock:
            self.conv1 = self.rep_block(self.filters, kernel_size=3)
            self.conv2 = self.rep_block(self.filters, kernel_size=3)
        else:
            self.conv1 = self.rep_block(self.filters, kernel_size=3, training=self.training)
            self.conv2 = self.rep_block(self.filters, kernel_size=3, training=self.training)
            
        if input_shape[-1] != self.filters:
            self.shortcut = False
        else:
            self.shortcut = True
            
        if self.scale_weight:
            with tf.init_scope():
                self.alpha = tf.Variable(name="BottleRep.alpha",
                                         initial_value=tf.ones((1,), dtype=tf.float32),
                                         trainable=True)
        else:
            self.alpha = 1.0
            
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut:
            return x + self.alpha * inputs
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "rep_block": self.rep_block,
            "scale_weight": self.scale_weight,
            "training": self.training
        })
        return config


class BottleRep3(CustomLayer):
    
    def __init__(self, filters, rep_block=RepVGGBlock, scale_weight=False, training=False, *args, **kwargs):
        super(BottleRep3, self).__init__(*args, **kwargs)
        self.filters      = filters
        self.rep_block    = rep_block
        self.scale_weight = scale_weight
        self.training     = training

    def build(self, input_shape):
        if self.rep_block == RepVGGBlock:
            self.conv1 = self.rep_block(self.filters, kernel_size=3, training=self.training)
            self.conv2 = self.rep_block(self.filters, kernel_size=3, training=self.training)
            self.conv3 = self.rep_block(self.filters, kernel_size=3, training=self.training)
        else:
            self.conv1 = self.rep_block(self.filters, kernel_size=3)
            self.conv2 = self.rep_block(self.filters, kernel_size=3)
            self.conv3 = self.rep_block(self.filters, kernel_size=3)
            
        if input_shape[-1] != self.filters:
            self.shortcut = False
        else:
            self.shortcut = True
            
        if self.scale_weight:
            with tf.init_scope():
                self.alpha = tf.Variable(name="BottleRep.alpha",
                                         initial_value=tf.ones((1,), dtype=tf.float32),
                                         trainable=True)
        else:
            self.alpha = 1.0
            
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        if self.shortcut:
            return x + self.alpha * inputs
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "rep_block": self.rep_block,
            "scale_weight": self.scale_weight,
            "training": self.training
        })
        return config


class RepBlock(CustomLayer):
    
    def __init__(self, filters, rep_block=RepVGGBlock, iters=1, training=False, *args, **kwargs):
        super(RepBlock, self).__init__(*args, **kwargs)
        self.filters   = filters
        self.rep_block = rep_block
        self.iters     = iters
        self.training  = training
        self.sub_block = kwargs.get('sub_block', RepVGGBlock)

    def build(self, input_shape):
        if self.rep_block != BottleRep:
            self.conv1 = self.rep_block(self.filters, kernel_size=3, training=self.training)
            self.block = Sequential([
                self.rep_block(self.filters, kernel_size=3, training=self.training) for i in range(self.iters - 1)
            ]) if self.iters > 1 else None
        else:
            self.conv1 = self.rep_block(self.filters, rep_block=self.sub_block, scale_weight=True, training=self.training)
            self.iters = self.iters // 2
            self.block = Sequential([
                self.rep_block(self.filters, rep_block=self.sub_block, scale_weight=True, training=self.training) for i in range(self.iters - 1)
            ]) if self.iters > 1 else None

        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        if self.block is not None:
            x = self.block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "rep_block": self.rep_block,
            "iters": self.iters,
            "training": self.training,
        })
        return config


class BepC3(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 rep_block=BottleRep,
                 sub_block=None,
                 expansion=0.5, 
                 iters=1, 
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 training=False, 
                 *args, 
                 **kwargs):
        super(BepC3, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.rep_block  = rep_block
        self.sub_block  = sub_block
        self.expansion  = expansion
        self.iters      = iters
        self.activation = activation
        self.normalizer = normalizer
        self.training   = training
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        if self.sub_block:
            self.block = RepBlock(hidden_dim, rep_block=self.rep_block, sub_block=self.sub_block, iters=self.iters, training=self.training)
        else:
            self.block = RepBlock(hidden_dim, rep_block=self.rep_block, iters=self.iters, training=self.training)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.block(x, training=training)
        y = self.conv2(inputs, training=training)
        out = concatenate([x, y], axis=-1)
        out = self.conv3(out, training=training)
        return out
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
            "iters": self.iters,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "training": self.training,
        })
        return config


class MBLABlock(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 rep_block=BottleRep3,
                 sub_block=None,
                 expansion=0.5, 
                 iters=1, 
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 training=False, 
                 *args, 
                 **kwargs):
        super(MBLABlock, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.rep_block  = rep_block
        self.sub_block  = sub_block if sub_block else RepVGGBlock
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.training   = training

        iters = iters // 2
        if iters <= 0:
            self.iters = 1
        else:
            self.iters = iters
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        if self.iters == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            while extra_branch_steps * 2 < self.iters:
                extra_branch_steps *= 2
            n_list = [0, extra_branch_steps, self.iters]
        self.branch_num = len(n_list)

        self.conv1 = ConvolutionBlock(self.branch_num * hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.block = []
        for n_list_i in n_list[1:]:
            self.block.append(
                [self.rep_block(hidden_dim, self.sub_block, scale_weight=True, training=self.training) for i in range(n_list_i)]
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = tf.split(x, num_or_size_splits=self.branch_num, axis=-1)

        merge_list = [x[0]]
        for m_idx, m_block in enumerate(self.block):
            merge_list.append(x[m_idx + 1])
            merge_list.extend(block(merge_list[-1]) for block in m_block)

        out = concatenate(merge_list, axis=-1)
        out = self.conv2(out, training=training)
        return out
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
            "iters": self.iters,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "training": self.training,
        })
        return config


class BiFusion(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 activation = 'relu', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(BiFusion, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.upsample = Conv2DTranspose(self.filters, 
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding='valid')
        self.downsample = ConvolutionBlock(self.filters, 3, downsample=True, activation=self.activation, normalizer=self.normalizer)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x0 = self.upsample(inputs[0], training=training)
        x1 = self.conv1(inputs[1], training=training)
        x2 = self.conv2(inputs[2], training=training)
        x2 = self.downsample(x2, training=training)
        out = concatenate([x0, x1], axis=-1)
        out = self.conv3(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "activation": self.activation,
            "normalizer": self.normalizer,
        })
        return config


class SEBlock(CustomLayer):
    
    def __init__(self, 
                 expansion  = 0.5,
                 activation = 'relu', 
                 normalizer = None, 
                 *args, 
                 **kwargs):
        super(SEBlock, self).__init__(*args, **kwargs)
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        bs = input_shape[-1]
        hidden_dim = int(bs * self.expansion)
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(bs, 1, activation='hard-sigmoid', normalizer=self.normalizer)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv1(x, training=training)        
        x = self.conv2(x, training=training)
        return x * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
        })
        return config


class Lite_EffiBlockS1(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 strides=(1, 1),
                 expansion = 1,
                 activation = 'hard-swish', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(Lite_EffiBlockS1, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.strides    = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv_pw = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv_dw = self.convolution_block(hidden_dim, 3, self.strides, normalizer=self.normalizer)
        self.se_block  = SEBlock(expansion=0.25)
        self.conv = ConvolutionBlock(self.filters // 2, 1, activation=self.activation, normalizer=self.normalizer)
        self.shuffle = ChannelShuffle(2)
        super().build(input_shape)

    def convolution_block(self, filters, kernel_size, strides, normalizer):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       groups=filters,
                       use_bias=False),
                get_normalizer_from_name(normalizer),
        ])

        
    def call(self, inputs, training=False):
        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        x2 = self.conv_pw(x2, training=training)
        x3 = self.conv_dw(x2, training=training)
        x3 = self.se_block(x3, training=training)
        x3 = self.conv(x3, training=training)
        out = concatenate([x1, x3], axis=-1)
        return self.shuffle(out)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
        })
        return config


class Lite_EffiBlockS2(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 strides=(1, 1),
                 expansion = 1,
                 activation = 'hard-swish', 
                 normalizer = 'batch-norm', 
                 *args, 
                 **kwargs):
        super(Lite_EffiBlockS2, self).__init__(*args, **kwargs)
        self.filters    = filters
        self.strides    = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.expansion  = expansion
        self.activation = activation
        self.normalizer = normalizer
        
    def build(self, input_shape):
        channel_dim = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        self.conv_dw_1 = self.convolution_block(channel_dim, 3, self.strides, normalizer=self.normalizer)
        self.conv_1 = ConvolutionBlock(self.filters // 2, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv_pw_2 = ConvolutionBlock(hidden_dim // 2, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv_dw_2 = self.convolution_block(hidden_dim // 2, 3, self.strides, normalizer=self.normalizer)
        self.se_block  = SEBlock(expansion=0.25)
        self.conv_2 = ConvolutionBlock(self.filters // 2, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv_pw_3 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv_dw_3 = ConvolutionBlock(self.filters, 3, groups=self.filters, activation=self.activation, normalizer=self.normalizer)
        super().build(input_shape)

    def convolution_block(self, filters, kernel_size, strides, normalizer):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       groups=filters,
                       use_bias=False),
                get_normalizer_from_name(normalizer),
        ])

    def call(self, inputs, training=False):
        x1 = self.conv_dw_1(inputs, training=training)
        x1 = self.conv_1(x1, training=training)
        x2 = self.conv_pw_2(inputs, training=training)
        x2 = self.conv_dw_2(x2, training=training)
        x2 = self.se_block(x2, training=training)
        x2 = self.conv_2(x2, training=training)

        out = concatenate([x1, x2], axis=-1)
        out = self.conv_dw_3(out, training=training)
        out = self.conv_pw_3(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
        })
        return config


class DPBlock(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 kernel_size = (3, 3),
                 strides     = (1, 1),
                 padding     = "same",
                 activation  = 'hard-swish', 
                 normalizer  = 'batch-norm', 
                 fuse        = False,
                 *args, 
                 **kwargs):
        super(DPBlock, self).__init__(*args, **kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides     = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding     = padding
        self.activation  = activation
        self.normalizer  = normalizer
        self.fuse        = fuse
        
    def build(self, input_shape):
        group = self.filters if input_shape[-1] % self.filters == 0 else 1
        self.conv_dw_1 = Conv2D(filters=self.filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                padding=self.padding,
                                groups=group)
        self.norm_1    = get_normalizer_from_name(self.normalizer)
        self.activ_1   = get_activation_from_name(self.activation)
        self.conv_pw_1 = Conv2D(filters=self.filters,
                                kernel_size=1,
                                strides=1,
                                padding="valid")
        self.norm_2    = get_normalizer_from_name(self.normalizer)
        self.activ_2   = get_activation_from_name(self.activation)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv_dw_1(inputs, training=training)
        if not self.fuse:
            x = self.norm_1(x, training=training)
        x = self.activ_1(x, training=training)
        x = self.conv_pw_1(x, training=training)
        if not self.fuse:
            x = self.norm_2(x, training=training)
        x = self.activ_2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "fuse": self.fuse
        })
        return config


class DarknetBlock(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 kernel_size = (3, 3),
                 expansion   = 0.5,
                 activation  = 'hard-swish', 
                 normalizer  = 'batch-norm', 
                 fuse        = False,
                 *args, 
                 **kwargs):
        super(DarknetBlock, self).__init__(*args, **kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.expansion   = expansion
        self.activation  = activation
        self.normalizer  = normalizer
        self.fuse        = fuse
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = DPBlock(self.filters, self.kernel_size, 1, "same", activation=self.activation, normalizer=self.normalizer, fuse=self.fuse)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "fuse": self.fuse
        })
        return config


class CSPBlock(CustomLayer):
    
    def __init__(self, 
                 filters, 
                 kernel_size = (3, 3),
                 expansion   = 0.5,
                 activation  = 'hard-swish', 
                 normalizer  = 'batch-norm', 
                 fuse        = False,
                 *args, 
                 **kwargs):
        super(CSPBlock, self).__init__(*args, **kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.expansion   = expansion
        self.activation  = activation
        self.normalizer  = normalizer
        self.fuse        = fuse
        
    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv2 = ConvolutionBlock(hidden_dim, 1, activation=self.activation, normalizer=self.normalizer)
        self.conv3 = ConvolutionBlock(self.filters, 1, activation=self.activation, normalizer=self.normalizer)
        self.block = DarknetBlock(hidden_dim, self.kernel_size, expansion=1.0, activation=self.activation, normalizer=self.normalizer, fuse=self.fuse)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x1 = self.block(x1, training=training)
        
        x2 = self.conv2(inputs, training=training)
        x = concatenate([x1, x2], axis=-1)
        x = self.conv3(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "fuse": self.fuse
        })
        return config


def EfficientRep(blocks=[RepVGGBlock, RepBlock],
                 filters=[64, 128, 256, 512, 1024],
                 layers=[6, 12, 18, 6],
                 use_csp=True,
                 csp_epsilon=1.0,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 activation='silu',
                 normalizer='batch-norm',
                 final_activation="softmax",
                 classes=1000,
                 training=False):
    
    '''
        EfficientRep Backbone
        EfficientRep is handcrafted by hardware-aware neural network design.
        With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''
    
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

    if use_csp:
        channel_merge_layer = CSPSPPF
    else:
        channel_merge_layer = SPPF

    if isinstance(blocks, (list, tuple)):
        block1, block2 = blocks
    else:
        block1 = blocks
        block2 = RepBlock

    if block1 == ConvolutionBlock:
        x = block1(filters=filters[0], kernel_size=3, downsample=True, activation=activation, normalizer=normalizer, name='stem')(img_input)
    else:
        x = block1(filters=filters[0], kernel_size=3, strides=2, training=training, name='stem')(img_input)

    for i in range(len(layers)):
        if block1 == ConvolutionBlock:
            x = block1(filters=filters[i + 1], kernel_size=3, downsample=True, activation=activation, normalizer=normalizer, name=f'stage{i + 1}.block1')(x)
        else:
            x = block1(filters=filters[i + 1], kernel_size=3, strides=2, training=training, name=f'stage{i + 1}.block1')(x)
        
        x = block2(filters=filters[i + 1], sub_block=block1, expansion=csp_epsilon, iters=layers[i], training=training, name=f'stage{i + 1}.block2')(x)

    x = channel_merge_layer(filters=filters[-1], 
                            pool_size=(5, 5),
                            expansion=0.5,
                            activation=activation, 
                            normalizer=normalizer,
                            name=f'stage{i + 1}.block3')(x)

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

    if block2 == MBLABlock:
        name = "MLBA"
    else:
        name = "Rep"
        
    if filters[0] == 16:
        suffit = "nano"
    elif filters[0] == 32:
        suffit = "small"
    elif filters[0] == 48:
        suffit = "medium"
    elif filters[0] == 64:
        suffit = "large"
    else:
        suffit = ""

    if len(filters) > 5:
        model = Model(inputs, x, name=f'Efficient-{name}{len(filters)}-{suffit}')
    else:
        model = Model(inputs, x, name=f'Efficient-{name}-{suffit}')

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


def EfficientRep_nano(blocks=[RepVGGBlock, RepBlock],
                      use_csp=True,
                      include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      activation='silu',
                      normalizer='batch-norm',
                      final_activation="softmax",
                      classes=1000,
                      training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[16, 32, 64, 128, 256],
                         layers=[2, 4, 6, 2],
                         use_csp=use_csp,
                         csp_epsilon=1.0,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep_nano_backbone(input_shape=(640, 640, 3),
                               include_top=False, 
                               weights='imagenet', 
                               activation='silu',
                               normalizer='batch-norm',
                               training=False,
                               custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6n
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n_finetune.py
    """
    
    model = EfficientRep_nano(include_top=include_top, 
                              weights=weights,
                              activation=activation,
                              normalizer=normalizer,
                              input_shape=input_shape,
                              training=training)

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


def EfficientRep6_nano(blocks=[RepVGGBlock, RepBlock],
                       use_csp=True,
                       include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       normalizer='batch-norm',
                       final_activation="softmax",
                       classes=1000,
                       training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[16, 32, 64, 128, 192, 256],
                         layers=[2, 4, 6, 2, 2],
                         use_csp=use_csp,
                         csp_epsilon=1.0,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep6_nano_backbone(input_shape=(1280, 1280, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='silu',
                                normalizer='batch-norm',
                                training=False,
                                custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6n6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n6_finetune.py
    """
    
    model = EfficientRep6_nano(include_top=include_top, 
                               weights=weights,
                               activation=activation,
                               normalizer=normalizer,
                               input_shape=input_shape,
                               training=training)

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
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def EfficientRep_small(blocks=[RepVGGBlock, RepBlock],
                       use_csp=True,
                       include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       normalizer='batch-norm',
                       final_activation="softmax",
                       classes=1000,
                       training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[32, 64, 128, 256, 512],
                         layers=[2, 4, 6, 2],
                         use_csp=use_csp,
                         csp_epsilon=1.0,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep_small_backbone(input_shape=(640, 640, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='silu',
                                normalizer='batch-norm',
                                training=False,
                                custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6s
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s_finetune.py
    """
    
    model = EfficientRep_small(include_top=include_top, 
                               weights=weights,
                               activation=activation,
                               normalizer=normalizer,
                               input_shape=input_shape,
                               training=training)

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


def EfficientRep6_small(blocks=[RepVGGBlock, RepBlock],
                        use_csp=True,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='silu',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000,
                        training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[32, 64, 128, 256, 384, 512],
                         layers=[2, 4, 6, 2, 2],
                         use_csp=use_csp,
                         csp_epsilon=1.0,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep6_small_backbone(input_shape=(1280, 1280, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='silu',
                                 normalizer='batch-norm',
                                 training=False,
                                 custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6s6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s6_finetune.py
    """
    
    model = EfficientRep6_small(include_top=include_top, 
                                weights=weights,
                                activation=activation,
                                normalizer=normalizer,
                                input_shape=input_shape,
                                training=training)

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
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def EfficientRep_medium(blocks=[RepVGGBlock, BepC3],
                        use_csp=False,
                        csp_epsilon=2/3,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='silu',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000,
                        training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[48, 96, 192, 384, 768],
                         layers=[4, 7, 11, 4],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep_medium_backbone(input_shape=(640, 640, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='silu',
                                 normalizer='batch-norm',
                                 training=False,
                                 custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6m
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m_finetune.py
    """
    
    model = EfficientRep_medium(include_top=include_top, 
                                weights=weights,
                                activation=activation,
                                normalizer=normalizer,
                                input_shape=input_shape,
                                training=training)

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


def EfficientRep6_medium(blocks=[RepVGGBlock, BepC3],
                         use_csp=False,
                         csp_epsilon=2/3,
                         include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         activation='silu',
                         normalizer='batch-norm',
                         final_activation="softmax",
                         classes=1000,
                         training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[48, 96, 192, 384, 576, 768],
                         layers=[4, 7, 11, 4, 4],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep6_medium_backbone(input_shape=(1280, 1280, 3),
                                  include_top=False, 
                                  weights='imagenet', 
                                  activation='silu',
                                  normalizer='batch-norm',
                                  training=False,
                                  custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6m6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m6_finetune.py
    """
    
    model = EfficientRep6_medium(include_top=include_top, 
                                 weights=weights,
                                 activation=activation,
                                 normalizer=normalizer,
                                 input_shape=input_shape,
                                 training=training)

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
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def EfficientRep_large(blocks=[ConvolutionBlock, BepC3],
                       use_csp=False,
                       csp_epsilon=1/2,
                       include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       normalizer='batch-norm',
                       final_activation="softmax",
                       classes=1000,
                       training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[64, 128, 256, 512, 1024],
                         layers=[6, 12, 18, 6],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep_large_backbone(input_shape=(640, 640, 3),
                                include_top=False, 
                                weights='imagenet', 
                                activation='silu',
                                normalizer='batch-norm',
                                training=False,
                                custom_layers=None) -> Model:

    """
        - Used in YOLOv6l
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l_finetune.py
    """

    model = EfficientRep_large(include_top=include_top, 
                               weights=weights,
                               activation=activation,
                               normalizer=normalizer,
                               input_shape=input_shape,
                               training=training)

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


def EfficientRep6_large(blocks=[ConvolutionBlock, BepC3],
                       use_csp=False,
                       csp_epsilon=1/2,
                       include_top=True,
                       weights='imagenet',
                       input_tensor=None,
                       input_shape=None,
                       pooling=None,
                       activation='silu',
                       normalizer='batch-norm',
                       final_activation="softmax",
                       classes=1000,
                       training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[64, 128, 256, 512, 768, 1024],
                         layers=[6, 12, 18, 6, 6],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientRep6_large_backbone(input_shape=(1280, 1280, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='silu',
                                 normalizer='batch-norm',
                                 training=False,
                                 custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6l6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l6_finetune.py
    """
    
    model = EfficientRep6_large(include_top=include_top, 
                                weights=weights,
                                activation=activation,
                                normalizer=normalizer,
                                input_shape=input_shape,
                                training=training)

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
        y_32 = model.get_layer("stage4.block2").output
        y_64 = model.get_layer("stage5.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_64], name=model.name + '_backbone')


def EfficientMBLA_small(blocks=[ConvolutionBlock, MBLABlock],
                        use_csp=False,
                        csp_epsilon=1/2,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='silu',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000,
                        training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[32, 64, 128, 256, 512],
                         layers=[2, 4, 4, 2],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientMBLA_small_backbone(input_shape=(640, 640, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='silu',
                                 normalizer='batch-norm',
                                 training=False,
                                 custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6 MBLA version small
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6s_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6s_mbla_finetune.py
    """
    
    model = EfficientMBLA_small(include_top=include_top, 
                                weights=weights,
                                activation=activation,
                                normalizer=normalizer,
                                input_shape=input_shape,
                                training=training)

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


def EfficientMBLA_medium(blocks=[ConvolutionBlock, MBLABlock],
                         use_csp=False,
                         csp_epsilon=1/2,
                         include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         activation='silu',
                         normalizer='batch-norm',
                         final_activation="softmax",
                         classes=1000,
                         training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[48, 96, 192, 384, 768],
                         layers=[2, 4, 4, 2],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientMBLA_medium_backbone(input_shape=(640, 640, 3),
                                  include_top=False, 
                                  weights='imagenet', 
                                  activation='silu',
                                  normalizer='batch-norm',
                                  training=False,
                                  custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6 MBLA version medium
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6m_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6m_mbla_finetune.py
    """
    
    model = EfficientMBLA_medium(include_top=include_top, 
                                 weights=weights,
                                 activation=activation,
                                 normalizer=normalizer,
                                 input_shape=input_shape,
                                 training=training)

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


def EfficientMBLA_large(blocks=[ConvolutionBlock, MBLABlock],
                        use_csp=False,
                        csp_epsilon=1/2,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='silu',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000,
                        training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[64, 128, 256, 512, 1024,],
                         layers=[2, 4, 4, 2],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientMBLA_large_backbone(input_shape=(640, 640, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='silu',
                                 normalizer='batch-norm',
                                 training=False,
                                 custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6 MBLA version large
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6l_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6l_mbla_finetune.py
    """
    
    model = EfficientMBLA_large(include_top=include_top, 
                                weights=weights,
                                activation=activation,
                                normalizer=normalizer,
                                input_shape=input_shape,
                                training=training)

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


def EfficientMBLA_xlarge(blocks=[ConvolutionBlock, MBLABlock],
                         use_csp=False,
                         csp_epsilon=1/2,
                         include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         activation='silu',
                         normalizer='batch-norm',
                         final_activation="softmax",
                         classes=1000,
                         training=False) -> Model:
    
    model = EfficientRep(blocks=blocks,
                         filters=[64, 128, 256, 512, 1024,],
                         layers=[4, 8, 8, 4],
                         use_csp=use_csp,
                         csp_epsilon=csp_epsilon,
                         include_top=include_top,
                         weights=weights, 
                         input_tensor=input_tensor, 
                         input_shape=input_shape, 
                         pooling=pooling, 
                         activation=activation,
                         normalizer=normalizer,
                         final_activation=final_activation,
                         classes=classes,
                         training=training)
    return model


def EfficientMBLA_xlarge_backbone(input_shape=(640, 640, 3),
                                  include_top=False, 
                                  weights='imagenet', 
                                  activation='silu',
                                  normalizer='batch-norm',
                                  training=False,
                                  custom_layers=None) -> Model:
    
    """
        - Used in YOLOv6 MBLA version xlarge
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6x_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6x_mbla_finetune.py
    """
    
    model = EfficientMBLA_xlarge(include_top=include_top, 
                                 weights=weights,
                                 activation=activation,
                                 normalizer=normalizer,
                                 input_shape=input_shape,
                                 training=training)

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


def EfficientLite(blocks=[Lite_EffiBlockS2, Lite_EffiBlockS1],
                  filters=[24, 32, 64, 128, 256],
                  filters_expansion=2,
                  layers=[1, 3, 7, 3],
                  include_top=True,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=None,
                  pooling=None,
                  activation='hard-swish',
                  normalizer='batch-norm',
                  final_activation="softmax",
                  classes=1000,
                  training=False):

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

    x = ConvolutionBlock(filters[0], 3, downsample=True, activation=activation, normalizer=normalizer, name='stem')(img_input)

    for layer_idx, layer in enumerate(layers):
        for idx in range(layer):
            if idx == 0:
                x = blocks[0](filters=filters[layer_idx + 1], strides=(2, 2), expansion=filters_expansion, name=f'stage{layer_idx + 1}.block{idx + 1}')(x)
            else:
                x = blocks[1](filters=filters[layer_idx + 1], strides=(1, 1), expansion=filters_expansion, name=f'stage{layer_idx + 1}.block{idx + 1}')(x)

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

    if filters == [24, 32, 48, 96, 176]:
        model = Model(inputs, x, name=f'Efficient-Lite-small')
    elif filters == [24, 32, 64, 144, 288]:
        model = Model(inputs, x, name=f'Efficient-Lite-medium')
    elif filters == [24, 48, 96, 192, 384]:
        model = Model(inputs, x, name=f'Efficient-Lite-large')
    else:
        model = Model(inputs, x, name=f'Efficient-Lite')
        
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


def EfficientLite_small(blocks=[Lite_EffiBlockS2, Lite_EffiBlockS1],
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='hard-swish',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000) -> Model:
    
    model = EfficientLite(blocks=blocks,
                          filters=[24, 32, 48, 96, 176],
                          filters_expansion=0.5,
                          layers=[1, 3, 7, 3],
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


def EfficientLite_small_backbone(input_shape=(640, 640, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='hard-swish',
                                 normalizer='batch-norm',
                                 custom_layers=None) -> Model:

    """
        - Used in YOLOv6 lite version small
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_s.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_s_finetune.py
    """

    model = EfficientLite_small(include_top=include_top, 
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
        y_4 = model.get_layer("stage1.block1").output
        y_8 = model.get_layer("stage2.block3").output
        y_16 = model.get_layer("stage3.block7").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def EfficientLite_medium(blocks=[Lite_EffiBlockS2, Lite_EffiBlockS1],
                         include_top=True,
                         weights='imagenet',
                         input_tensor=None,
                         input_shape=None,
                         pooling=None,
                         activation='hard-swish',
                         normalizer='batch-norm',
                         final_activation="softmax",
                         classes=1000) -> Model:
    
    model = EfficientLite(blocks=blocks,
                          filters=[24, 32, 64, 144, 288],
                          filters_expansion=0.5,
                          layers=[1, 3, 7, 3],
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


def EfficientLite_medium_backbone(input_shape=(640, 640, 3),
                                  include_top=False, 
                                  weights='imagenet', 
                                  activation='hard-swish',
                                  normalizer='batch-norm',
                                  custom_layers=None) -> Model:

    """
        - Used in YOLOv6 lite version medium
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_m.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_m_finetune.py
    """

    model = EfficientLite_medium(include_top=include_top, 
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
        y_4 = model.get_layer("stage1.block1").output
        y_8 = model.get_layer("stage2.block3").output
        y_16 = model.get_layer("stage3.block7").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')


def EfficientLite_large(blocks=[Lite_EffiBlockS2, Lite_EffiBlockS1],
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        activation='hard-swish',
                        normalizer='batch-norm',
                        final_activation="softmax",
                        classes=1000) -> Model:
    
    model = EfficientLite(blocks=blocks,
                          filters=[24, 48, 96, 192, 384],
                          filters_expansion=0.5,
                          layers=[1, 3, 7, 3],
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


def EfficientLite_large_backbone(input_shape=(640, 640, 3),
                                 include_top=False, 
                                 weights='imagenet', 
                                 activation='hard-swish',
                                 normalizer='batch-norm',
                                 custom_layers=None) -> Model:

    """
        - Used in YOLOv6 lite version large
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_l.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_l_finetune.py
    """

    model = EfficientLite_large(include_top=include_top, 
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
        y_4 = model.get_layer("stage1.block1").output
        y_8 = model.get_layer("stage2.block3").output
        y_16 = model.get_layer("stage3.block7").output
        y_32 = model.get_layer("stage4.block3").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')