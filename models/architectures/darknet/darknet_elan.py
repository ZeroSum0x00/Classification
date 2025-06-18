"""
    DarknetELAN: YOLOv7 Backbone with ELAN, TransitionBlock, and ScaleUpConcatBlock
    
    Overview:
        DarknetELAN implements the backbone and neck architecture used in YOLOv7, 
        designed for high-performance object detection with efficiency and scalability. 
        It introduces modular components that enhance gradient flow, reuse features,
        and support multi-scale feature aggregation.
    
    General Model Architecture:
        - Darknet ELAN A (tiny):
             --------------------------------------------------------------------------------
            | Stage                  | Layer                       | Output Shape            |
            |------------------------+-----------------------------+-------------------------|
            | Input                  | input_layer                 | (None, 640, 640, 3)     |
            |------------------------+-----------------------------+-------------------------|
            | Stem                   | ConvolutionBlock (3x3, s=2) | (None, 320, 320, C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 160, 160, 2C)    |
            |                        | ConvolutionBlock (3x3, s=1) | (None, 160, 160, C)     |
            |                        | ScaleUpConcatBlock          | (None, 160, 160, 2C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 2                | MaxPooling2D                | (None, 80, 80, 2C)      |
            |                        | ConvolutionBlock (3x3, s=1) | (None, 80, 80, 2C)      |
            |                        | ScaleUpConcatBlock          | (None, 80, 80, 4C)      |
            |------------------------+-----------------------------+-------------------------|
            | Stage 3                | MaxPooling2D                | (None, 40, 40, 4C)      |
            |                        | Conv3 (3x3, s=1)            | (None, 40, 40, 4C)      |
            |                        | ScaleUpConcatBlock          | (None, 40, 40, 8C)      |
            |------------------------+-----------------------------+-------------------------|
            | Stage 4                | MaxPooling2D                | (None, 20, 20, 8C)      |
            |                        | Conv3 (3x3, s=1)            | (None, 20, 20, 8C)      |
            |                        | ScaleUpConcatBlock          | (None, 20, 20, 16C*S)   |
            |                        | pyramid_poolings (*)        | (None, 20, 20, 16C*S)   |
            |------------------------+-----------------------------+-------------------------|
            | CLS Logics             | GlobalAveragePooling        | (None, 16C*S)           |
            |                        | fc (Logics)                 | (None, 1000)            |
             --------------------------------------------------------------------------------
             
        - Darknet ELAN B (nano, small):
             --------------------------------------------------------------------------------
            | Stage                  | Layer                       | Output Shape            |
            |------------------------+-----------------------------+-------------------------|
            | Input                  | input_layer                 | (None, 640, 640, 3)     |
            |------------------------+-----------------------------+-------------------------|
            | Stem                   | ConvolutionBlock (3x3, s=1) | (None, 640, 640, C)     |
            |                        | ConvolutionBlock (3x3, s=2) | (None, 320, 320, 2C)    |
            |                        | ConvolutionBlock (3x3, s=1) | (None, 320, 320, 2C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 160, 160, 4C)    |
            |                        | ScaleUpConcatBlock          | (None, 160, 160, 8C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 2                | TransitionBlock             | (None, 80, 80, 8C)      |
            |                        | ScaleUpConcatBlock          | (None, 80, 80, 16C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 3                | TransitionBlock             | (None, 40, 40, 16C)     |
            |                        | ScaleUpConcatBlock          | (None, 40, 40, 32C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 4                | TransitionBlock             | (None, 20, 20, 32C*S)   |
            |                        | ScaleUpConcatBlock          | (None, 20, 20, 32C*S)   |
            |                        | pyramid_poolings (*)        | (None, 20, 20, 32C*S)   |
            |------------------------+-----------------------------+-------------------------|
            | CLS Logics             | GlobalAveragePooling        | (None, 32C*S)           |
            |                        | fc (Logics)                 | (None, 1000)            |
             --------------------------------------------------------------------------------

        - Darknet ELAN C (medium, large, xlarge):
             --------------------------------------------------------------------------------
            | Stage                  | Layer                       | Output Shape            |
            |---------------------------------------------------------------------------|
            | Input                  | input_layer                 | (None, 1280, 1280, 3)   |
            |------------------------+-----------------------------+-------------------------|
            | Stem                   | ReOrg                       | (None, 640, 640, 12)    |
            |                        | ConvolutionBlock (3x3, s=1) | (None, 640, 640, C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 1                | DownC                       | (None, 320, 320, 2C)    |
            |                        | ScaleUpConcatBlock          | (None, 320, 320, 2C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 2                | DownC                       | (None, 160, 160, 4C)    |
            |                        | ScaleUpConcatBlock          | (None, 160, 160, 4C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 3                | DownC                       | (None, 80, 80, 8C)      |
            |                        | ScaleUpConcatBlock          | (None, 80, 80, 8C)      |
            |------------------------+-----------------------------+-------------------------|
            | Stage 4                | DownC                       | (None, 40, 40, 12C)     |
            |                        | ScaleUpConcatBlock          | (None, 40, 40, 12C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 4                | DownC                       | (None, 20, 20, 16C*S)   |
            |                        | ScaleUpConcatBlock          | (None, 20, 20, 16C*S)   |
            |                        | pyramid_poolings (*)        | (None, 20, 20, 16C*S)   |
            |------------------------+-----------------------------+-------------------------|
            | CLS Logics             | GlobalAveragePooling        | (None, 16C*S)           |
            |                        | fc (Logics)                 | (None, 1000)            |
             --------------------------------------------------------------------------------

        - Darknet ELAN D (huge):
             --------------------------------------------------------------------------------
            | Stage                  | Layer                       | Output Shape            |
            |------------------------+-----------------------------+-------------------------|
            | Input                  | input_layer                 | (None, 1280, 1280, 3)   |
            |------------------------+-----------------------------+-------------------------|
            | Stem                   | ReOrg                       | (None, 640, 640, 12)    |
            |                        | ConvolutionBlock (3x3, s=1) | (None, 640, 640, C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 1                | DownC                       | (None, 320, 320, 2C)    |
            |                        | ScaleUpConcatBlock          | (None, 320, 320, 2C)    |
            |                        | ScaleUpConcatBlock          | (None, 320, 320, 2C)    |
            |                        | Shortcut                    | (None, 320, 320, 2C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 2                | DownC                       | (None, 160, 160, 4C)    |
            |                        | ScaleUpConcatBlock          | (None, 160, 160, 4C)    |
            |                        | ScaleUpConcatBlock          | (None, 160, 160, 4C)    |
            |                        | Shortcut                    | (None, 160, 160, 4C)    |
            |------------------------+-----------------------------+-------------------------|
            | Stage 3                | DownC                       | (None, 80, 80, 8C)      |
            |                        | ScaleUpConcatBlock          | (None, 80, 80, 8C)      |
            |                        | ScaleUpConcatBlock          | (None, 80, 80, 8C)      |
            |                        | Shortcut                    | (None, 80, 80, 8C)      |
            |------------------------+-----------------------------+-------------------------|
            | Stage 4                | DownC                       | (None, 40, 40, 12C)     |
            |                        | ScaleUpConcatBlock          | (None, 40, 40, 12C)     |
            |                        | ScaleUpConcatBlock          | (None, 40, 40, 12C)     |
            |                        | Shortcut                    | (None, 40, 40, 12C)     |
            |------------------------+-----------------------------+-------------------------|
            | Stage 5                | DownC                       | (None, 20, 20, 16C*S)   |
            |                        | ScaleUpConcatBlock          | (None, 20, 20, 16C*S)   |
            |                        | ScaleUpConcatBlock          | (None, 20, 20, 16C*S)   |
            |                        | Shortcut                    | (None, 20, 20, 16C*S)   |
            |                        | pyramid_poolings (*)        | (None, 20, 20, 16C*S)   |
            |------------------------+-----------------------------+-------------------------|
            | CLS Logics             | GlobalAveragePooling        | (None, 16C*S)           |
            |                        | fc (Logics)                 | (None, 1000)            |
             --------------------------------------------------------------------------------
        (*) Note: While the original architecture does not include a Pyramid Pooling layer, 
        it can be optionally incorporated to enhance feature aggregation and create an extended variant of the model.

    Model Parameter Comparison:
         --------------------------------------------
        |      Model Name          |    Params       |
        |--------------------------+-----------------|
        |    DarkNetELAN tiny      |    3,071,304    |
        |--------------------------+-----------------|
        |    DarkNetELAN nano      |   14,416,840    |
        |--------------------------+-----------------|
        |    DarkNetELAN small     |   21,582,016    |
        |--------------------------+-----------------|
        |    DarkNetELAN medium    |   37,040,616    |
        |--------------------------+-----------------|
        |    DarkNetELAN large     |   48,830,632    |
        |--------------------------+-----------------|
        |    DarkNetELAN xlarge    |   66,540,136    |
        |--------------------------+-----------------|
        |    DarkNetELAN huge      |   84,323,624    |
         --------------------------------------------

    Reference:
        - Paper: "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
          https://arxiv.org/pdf/2207.02696
          
        - Original implementation:
          https://github.com/WongKinYiu/yolov7
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Dense,
    Dropout, MaxPooling2D, GlobalAveragePooling2D,
    Concatenate, concatenate, add,
)

from .darknet19 import ConvolutionBlock
from .darknet_c3 import Bottleneck, GhostConv, GhostBottleneck
from .efficient_rep import CSPSPPF, RepVGGBlock

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    ScaleWeight, LinearLayer,
)
from utils.model_processing import (
    process_model_input, create_model_backbone, create_layer_instance,
    validate_conv_arg, check_regularizer,
)



class ReOrg(tf.keras.layers.Layer):
    
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    def call(self, inputs):
        x1 = inputs[:, ::2, ::2, :]
        x2 = inputs[:, 1::2, ::2, :]
        x3 = inputs[:, ::2, 1::2, :]
        x4 = inputs[:, 1::2, 1::2, :]
        x = concatenate([x1, x2, x3, x4], axis=self.axis)        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class ChunCat(tf.keras.layers.Layer):
    
    def __init__(self, chun_dim=2, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chun_dim = chun_dim
        self.axis = axis
        self.merger = Concatenate(axis=self.axis)
        
    def call(self, inputs):
        x1 = []
        x2 = []
        for input in inputs:
            i1, i2 = tf.split(
                input,
                num_or_size_splits=[self.chun_dim, inputs.shape[self.axis] - self.chun_dim],
                axis=self.axis
            )
            x1.append(i1)
            x2.append(i2)
        x = self.merger(x1 + x2)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "chun_dim": self.chun_dim,
            "axis": self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class Shortcut(tf.keras.layers.Layer):
    
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        
    def call(self, inputs, shortcut=None):
        if shortcut is not None:
            return inputs + shortcut
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FoldCut(tf.keras.layers.Layer):
    
    def __init__(self, fold_dim=2, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_dim = fold_dim
        self.axis = axis
        
    def call(self, inputs):
        x1, x2 = tf.split(
            inputs,
            num_or_size_splits=[self.fold_dim, inputs.shape[self.axis] - self.fold_dim],
            axis=self.axis
        )
        return x1 + x2

    def get_config(self):
        config = super().get_config()
        config.update({
            "fold_dim": self.fold_dim,
            "axis": self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class ImplicitAdd(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, stddev=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev
                     
    def build(self, input_shape):
        init_value = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        self.implicit = tf.Variable(
            name="implicit",
            initial_value=init_value(shape=(1, 1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True)
        
    def call(self, inputs, training=False):
        return inputs + self.implicit

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.mean,
            "stddev": self.stddev
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class ImplicitMul(tf.keras.layers.Layer):
    def __init__(self, mean=1.0, stddev=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev
                     
    def build(self, input_shape):
        init_value = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        
        self.implicit = tf.Variable(
            name="implicit",
            initial_value=init_value(shape=(1, 1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs, training=False):
        return inputs * self.implicit

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.mean,
            "stddev": self.stddev
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RobustConv(tf.keras.layers.Layer):
    
    """
        Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        groups=1,
        conv_scale_init=1.0,
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
        self.kernel_size = validate_conv_arg(kernel_size)
        self.strides = validate_conv_arg(strides)
        self.padding = padding
        self.groups = groups
        self.conv_scale_init = conv_scale_init
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
    
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        
        self.conv_dw  = Sequential([
                Conv2D(
                    filters=hidden_dim,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    groups=hidden_dim,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                ),
                get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
                get_activation_from_name(self.activation)
        ])
        
        self.conv_1x1 = Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        if self.conv_scale_init > 0:
            self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
            
        super().build(input_shape)

    
    def call(self, inputs, training=False):
        x = self.conv_dw(inputs, training=training)
        x = self.conv_1x1(x, training=training)
        if hasattr(self, "scale_layer"):
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
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config


class RobustConv2(RobustConv):
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        groups=1,
        conv_scale_init=1.0,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        kernel_size = validate_conv_arg(kernel_size)
        strides = validate_conv_arg(strides)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            conv_scale_init=conv_scale_init,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        
        self.conv_deconv = Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.strides,
            strides=self.strides,
            padding="valid",
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        if self.conv_scale_init > 0:
            self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
        
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv_dw(inputs, training=training)
        x = self.conv_deconv(x, training=training)
        
        if hasattr(self, "scale_layer"):
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
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config


class BasicStem(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        pool_size=(2, 2),
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
        self.pool_size = validate_conv_arg(pool_size)
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        hidden_dim = self.filters // 2
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
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
        
        self.conv3 = ConvolutionBlock(
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv4 = ConvolutionBlock(
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
        
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2))

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class GhostStem(BasicStem):

    def __init__(
        self,
        filters,
        pool_size=(2, 2),
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        pool_size = validate_conv_arg(pool_size)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            pool_size=pool_size,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
            
    def build(self, input_shape):
        hidden_dim = self.filters // 2
        
        self.conv1 = GhostConv(
            filters=hidden_dim,
            kernel_size=(3, 3),
            downsample=True,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = GhostConv(
            filters=hidden_dim,
            kernel_size=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv3 = GhostConv(
            filters=hidden_dim,
            kernel_size=(3, 3),
            downsample=True,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv4 = GhostConv(
            filters=self.filters,
            kernel_size=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(2, 2))

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class DownC(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        pool_size=(2, 2),
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
        self.pool_size = validate_conv_arg(pool_size)
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim1 = input_shape[-1]
        hidden_dim2 = self.filters // 2
        
        self.conv1 = ConvolutionBlock(
            filters=hidden_dim1,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = Sequential([
            Conv2D(
                filters=hidden_dim2,
                kernel_size=(3, 3),
                strides=self.pool_size,
                padding="same",
                groups=1,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name(self.activation)
        ])
        
        self.conv3 = ConvolutionBlock(
            filters=hidden_dim2,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=self.pool_size)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        y = self.pool(inputs)
        y = self.conv3(y, training=training)
        out = concatenate([x, y], axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class ResX(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        groups=1,
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
        self.groups = groups
        self.expansion = expansion
        self.shortcut = shortcut       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.c     = input_shape[-1]
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
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv3 = ConvolutionBlock(
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
        x = self.conv2(x, training=training)        
        x = self.conv3(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "groups": self.groups,
            "expansion": self.expansion,
            "shortcut": self.shortcut,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class CSPSPPC(CSPSPPF):

    def __init__(
        self,
        filters,
        pool_size=(5, 5),
        expansion=0.5,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        pool_size = validate_conv_arg(pool_size)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            pool_size=pool_size,
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        hidden_dim = int(2 * self.filters * self.expansion)
        super().build(input_shape)


class GhostCSPSPPC(CSPSPPC):

    def __init__(
        self,
        filters,
        pool_size=(5, 5),
        expansion=0.5,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        pool_size = validate_conv_arg(pool_size)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            pool_size=pool_size,
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )

    def build(self, input_shape):
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
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv3 = GhostConv(
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
        
        self.conv4 = GhostConv(
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
        
        self.conv5 = GhostConv(
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv6 = GhostConv(
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
        

        self.shortcut = GhostConv(
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

        self.pool  = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding="same")


class BottleneckCSPA(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        groups=1,
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
        self.groups = groups
        self.iters = iters
        self.expansion = expansion
        self.shortcut = shortcut       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
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
        
        self.conv3 = ConvolutionBlock(
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
        
        self.block = Sequential([
            Bottleneck(
                filters=hidden_dim,
                groups=self.groups,
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
            "groups": self.groups,
            "iters": self.iters,
            "expansion": self.expansion,
            "shortcut": self.shortcut,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class BottleneckCSPB(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        
        x1 = self.block(x, training=training)
        x2 = self.conv2(x, training=training)
        
        x = concatenate([x1, x2], axis=-1)
        x = self.conv3(x, training=training)
        return x


class BottleneckCSPC(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        self.conv3 = ConvolutionBlock(
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
        
        self.conv4 = ConvolutionBlock(
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
        x1 = self.conv1(inputs, training=training)
        x1 = self.block(x1, training=training)
        x1 = self.conv3(x1, training=training)

        x2 = self.conv2(inputs, training=training)
        
        x = concatenate([x1, x2], axis=-1)
        x = self.conv4(x, training=training)
        return x


class ResCSPA(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
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


class ResXCSPA(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=32,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
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


class GhostCSPA(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
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


class ResCSPB(BottleneckCSPB):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ) for _ in range(self.iters)
        ])


class ResXCSPB(BottleneckCSPB):
    def __init__(
        self,
        filters,
        groups=32,
        iters=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ) for _ in range(self.iters)
        ])


class GhostCSPB(BottleneckCSPB):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
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
            ) for _ in range(self.iters)
        ])


class ResCSPC(BottleneckCSPC):
    def __init__(
        self,
        filters,
        groups=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ) for _ in range(self.iters)
        ])


class ResXCSPC(BottleneckCSPC):
    def __init__(
        self,
        filters,
        groups=32,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            ResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            ) for _ in range(self.iters)
        ])


class GhostCSPC(BottleneckCSPC):
    def __init__(
        self,
        filters,
        groups=1,
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
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
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
            ) for _ in range(self.iters)
        ])


class RepBottleneck(Bottleneck):
    def __init__(
        self,
        filters,
        strides=(1, 1),
        groups=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        strides = validate_conv_arg(strides)
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            strides=strides,
            groups=groups,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        self.conv2 = RepVGGBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=self.strides,
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            deploy=self.deploy,
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RepBottleneckCSPA(BottleneckCSPA):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepBottleneck(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepBottleneckCSPB(BottleneckCSPB):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=1,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepBottleneck(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepBottleneckCSPC(BottleneckCSPC):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepBottleneck(
                filters=hidden_dim,
                groups=self.groups,
                expansion=1,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepRes(ResX):
    def __init__(
        self,
        filters,
        groups=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        
        self.conv2 = RepVGGBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            deploy=self.deploy,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResCSPA(ResCSPA):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepRes(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResCSPB(ResCSPB):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=1,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepRes(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResCSPC(ResCSPC):
    def __init__(
        self,
        filters,
        groups=1,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepRes(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResX(ResX):
    def __init__(
        self,
        filters,
        groups=32,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        
        self.conv2 = RepVGGBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            deploy=self.deploy,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RepResXCSPA(ResXCSPA):
    def __init__(
        self,
        filters,
        groups=32,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResXCSPB(ResXCSPB):
    def __init__(
        self,
        filters,
        groups=32,
        iters=1,
        expansion=1,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class RepResXCSPC(ResXCSPC):
    def __init__(
        self,
        filters,
        groups=32,
        iters=1,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        regularizer_decay = check_regularizer(regularizer_decay)
        
        super().__init__(
            filters=filters,
            groups=groups,
            iters=iters,
            expansion=expansion,
            shortcut=shortcut,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.deploy = deploy
        
    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepResX(
                filters=hidden_dim,
                groups=self.groups,
                expansion=0.5,
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            for _ in range(self.iters)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class ScaleUpConcatBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        iters=1,
        id_concat=[-1, -3, -5, -6],
        activation="silu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_filters, self.out_filters = filters
        self.iters = iters
        self.id_concat = id_concat       
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
                     
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(
            filters=self.in_filters,
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
            filters=self.in_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.blocks = [
            ConvolutionBlock(
                filters=self.in_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ]
        
        self.conv3 = ConvolutionBlock(
            filters=self.out_filters,
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
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)

        x = [x1, x2]
        for block in self.blocks:
            x2 = block(x2, training=training)
            x.append(x2)

        x = concatenate([x[i] for i in self.id_concat], axis=-1)
        x = self.conv3(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "iters": self.iters,
            "id_concat": self.id_concat,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
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
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
           
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(
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
        
        self.conv3 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        
    def call(self, inputs, training=False):
        x1 = self.pool(inputs)
        x1 = self.conv1(x1, training=training)
        
        x2 = self.conv2(inputs, training=training)
        x2 = self.conv3(x2, training=training)

        x = concatenate([x1, x2], axis=-1)
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def DarkNetELAN_A(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    regularizer_decay = check_regularizer(regularizer_decay)
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
        default_size=[640, 1280],
        min_size=32,
        weights=weights
    )

    if isinstance(filters, (tuple, list)):
        f0, f1 = filters
    else:
        f0 = filters
        f1 = filters * 2
        
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

    f0 = f0 if isinstance(f0, (tuple, list)) else [f0 * channel_scale**i for i in range(len(num_blocks))]
    f1 = f1 if isinstance(f1, (tuple, list)) else [f1 * channel_scale**i for i in range(len(num_blocks))]
    
    x = inputs
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            extractor_block1,
            filters=f0[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        fx, fy, fz = f0[i], f0[i + 1], f1[i]
        
        if is_last_stage:
            fx = int(fx * final_channel_scale)
            fy = int(fy * final_channel_scale)
            fz = int(fz * final_channel_scale)
            final_filters = fy
            
        if num_block > 0:
            if i == 0:
                x = create_layer_instance(
                    extractor_block1,
                    filters=fy,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    **layer_constant_dict,
                    name=f"{block_name_prefix}.block1"
                )(x)
            else:
                 x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"{block_name_prefix}.block1")(x)

        if num_block > 1:
            x = create_layer_instance(
                extractor_block2,
                filters=fx,
                kernel_size=(1, 1),
                strides=(1, 1),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block2"
            )(x)

        if num_block > 2:
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=[fz, fy],
                iters=num_block - 2,
                id_concat=id_concat,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block3"
            )(x)

    block_name_prefix = f"stage{len(num_blocks) - 1}"

    if final_filters is None:
        final_filters = int(f0[-1] * final_channel_scale)
        
    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{p + 4}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block4")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "DarkNet-ELAN"
    if filters == [32, 32] and num_blocks == [1, 4, 4, 4, 4]:
        model_name += "-tiny"
    else:
        model_name += "-A"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetELAN_A_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block{j}" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_A,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_B(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
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

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    regularizer_decay = check_regularizer(regularizer_decay)
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
        default_size=[640, 1280],
        min_size=32,
        weights=weights
    )

    if isinstance(filters, (tuple, list)):
        f0, f1 = filters
    else:
        f0 = filters
        f1 = filters * 2
        
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

    f0 = [f0 * channel_scale**i for i in range(len(num_blocks) + 1)]
    f1 = [f1 * channel_scale**i for i in range(len(num_blocks) - 1)]
    
    x = inputs
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            extractor_block1,
            filters=f0[0] if i == 0 else f0[1],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 1 else (1, 1),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        fx = f0[i + 2] if i == 0 else f0[i + 1]
        
        if is_last_stage:
            fx = int(fx * final_channel_scale)
            fy = [f1[i - 1], int(f0[i + 2] * final_channel_scale)]
            final_filters = fx
        else:
            fy = [f1[i], f0[i + 3]]

        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i == 0 else extractor_block2,
                filters=fx,
                kernel_size=(3, 3),
                strides=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)

        if num_block > 1:
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=fy,
                iters=num_block - 1,
                id_concat=id_concat,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block2"
            )(x)

    block_name_prefix = f"stage{len(num_blocks) - 1}"
    
    if final_filters is None:
        final_filters = int(filters[-1] * final_channel_scale)
        
    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{p + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block3")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "DarkNet-ELAN"
    if filters == [32, 64] and num_blocks == [3, 5, 5, 5, 5]:
        model_name += "-nano"
    elif filters == [40, 64] and num_blocks == [3, 7, 7, 7, 7]:
        model_name += "-small"
    else:
        model_name += "-B"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetELAN_B_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block{j}" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]
    
    return create_model_backbone(
        model_fn=DarkNetELAN_B,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_C(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
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

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    regularizer_decay = check_regularizer(regularizer_decay)
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
        default_size=[640, 1280],
        min_size=32,
        weights=weights
    )

    if isinstance(filters, (tuple, list)):
        f0, f1 = filters
    else:
        f0 = filters
        f1 = filters * 2
        
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

    f0 = f0 if isinstance(f0, (tuple, list)) else [f0 * channel_scale**i for i in range(len(num_blocks))]
    f1 = f1 if isinstance(f1, (tuple, list)) else [f1 * channel_scale**i for i in range(len(num_blocks) - 1)]

    x = inputs
    for i in range(num_blocks[0]):
        if i == 0:
            x = ReOrg(name=f"stem.block{i + 1}")(x)
        else:
            x = ConvolutionBlock(
                filters=f0[0],
                kernel_size=(3, 3),
                strides=(1, 1),
                **layer_constant_dict,
                name=f"stem.block{i + 1}"
            )(x)
    
    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        fx = f0[i + 1]
        fy = f1[i]
        
        if is_last_stage:
            fx -= f0[i]
            fy -= f1[i - 1]

            fx = int(fx * final_channel_scale)
            fy = int(fy * final_channel_scale)
            final_filters = fx
        elif i == len(num_blocks[1:]) - 2:
            fx -= f0[i - 1]
            fy -= f1[i - 2]

        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i < 1 else extractor_block2,
                filters=fx,
                kernel_size=(3, 3),
                strides=(2, 2),
                pool_size=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)

        if num_block > 1:
            x = create_layer_instance(
                fusion_block1 if i < 1 else fusion_block2,
                filters=[fy, fx],
                iters=num_block - 1,
                id_concat=id_concat,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block2"
            )(x)
            
    block_name_prefix = f"stage{len(num_blocks) - 1}"

    if final_filters is None:
        final_filters = int((f0[-1] - f0[-2]) * final_channel_scale)
        
    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{p + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block3")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "DarkNet-ELAN"
    if filters == [64, 64] and num_blocks == [2, 5, 5, 5, 5, 5]:
        model_name += "-medium"
    elif filters == [80, 64] and num_blocks == [2, 7, 7, 7, 7, 7]:
        model_name += "-large"
    elif filters == [96, 64] and num_blocks == [2, 9, 9, 9, 9, 9]:
        model_name += "-xlarge"
    else:
        model_name += "-C"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetELAN_C_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block{j}" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_C,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_D(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    regularizer_decay = check_regularizer(regularizer_decay)
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
        default_size=[640, 1280],
        min_size=32,
        weights=weights
    )
    
    if isinstance(filters, (tuple, list)):
        f0, f1 = filters
    else:
        f0 = filters
        f1 = filters * 2
        
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

    f0 = f0 if isinstance(f0, (tuple, list)) else [f0 * channel_scale**i for i in range(len(num_blocks))]
    f1 = f1 if isinstance(f1, (tuple, list)) else [f1 * channel_scale**i for i in range(len(num_blocks) - 1)]

    x = inputs
    for i in range(num_blocks[0]):
        if i == 0:
            x = ReOrg(name=f"stem.block{i + 1}")(x)
        else:
            x = ConvolutionBlock(
                filters=f0[0],
                kernel_size=(3, 3),
                strides=(1, 1),
                **layer_constant_dict,
                name=f"stem.block{i + 1}"
            )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        fx = f0[i + 1]
        fy = f1[i]
        
        if is_last_stage:
            fx -= f0[i]
            fy -= f1[i - 1]

            fx = int(fx * final_channel_scale)
            fy = int(fy * final_channel_scale)
            final_filters = fx
        elif i == len(num_blocks[1:]) - 2:
            fx -= f0[i - 1]
            fy -= f1[i - 2]

        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i < 1 else extractor_block2,
                filters=fx,
                kernel_size=(3, 3),
                strides=(2, 2),
                pool_size=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)
            
        if num_block > 1:
            x1 = create_layer_instance(
                fusion_block1 if i < 1 else fusion_block2,
                filters=[fy, fx],
                iters=num_block - 1,
                id_concat=id_concat,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block2.branch1"
            )(x)

            x2 = create_layer_instance(
                fusion_block1 if i < 1 else fusion_block2,
                filters=[fy, fx],
                iters=num_block - 1,
                id_concat=id_concat,
                **layer_constant_dict,
                name=f"stage{i + 1}.block2.branch2"
            )(x1)
            
            x = Shortcut(name=f"stage{i + 1}.block2.merger")(x1, x2)

    block_name_prefix = f"stage{len(num_blocks) - 1}"
    
    if final_filters is None:
        final_filters = int(filters[-1] * final_channel_scale)
        
    if pyramid_pooling:
        for p, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=final_filters,
                **layer_constant_dict,
                name=f"{block_name_prefix}.block{p + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"{block_name_prefix}.block3")(x)
        
    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "DarkNet-ELAN"
    if filters == [80, 64] and num_blocks == [2, 7, 7, 7, 7, 7]:
        model_name += "-huge"
    else:
        model_name += "-D"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetELAN_D_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    id_concat,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        f"stem.block{j}" if i == 0 else f"stage{i}.block2.merger"
        for i, j in enumerate(num_blocks[:-1])
    ]
    
    return create_model_backbone(
        model_fn=DarkNetELAN_D,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_tiny(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_A(
        feature_extractor=ConvolutionBlock,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[32, 32],
        num_blocks=[1, 4, 4, 4, 4],
        id_concat=[-1, -2, -3, -4],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_tiny_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv7 tiny
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-tiny.yaml
    """

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2",
        "stage3.block3",
        "stage4.block3",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_tiny,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_nano(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_B(
        feature_extractor=[ConvolutionBlock, TransitionBlock],
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[32, 64],
        num_blocks=[3, 5, 5, 5, 5],
        id_concat=[-1, -3, -5, -6],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7.yaml
    """

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_nano,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_small(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_B(
        feature_extractor=[ConvolutionBlock, TransitionBlock],
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[40, 64],
        num_blocks=[3, 7, 7, 7, 7],
        id_concat=[-1, -3, -5, -7, -8],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7X
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7x.yaml
    """

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]
    
    return create_model_backbone(
        model_fn=DarkNetELAN_small,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_medium(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_C(
        feature_extractor=ConvolutionBlock,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[64, 64],
        num_blocks=[2, 5, 5, 5, 5, 5],
        id_concat=[-1, -3, -5, -6],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7-W6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-w6.yaml
    """

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_medium,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_large(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_C(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[80, 64],
        num_blocks=[2, 7, 7, 7, 7, 7],
        id_concat=[-1, -3, -5, -7, -8],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7-E6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/training/yolov7-e6.yaml
    """

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_xlarge(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_C(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[96, 64],
        num_blocks=[2, 9, 9, 9, 9, 9],
        id_concat=[-1, -3, -5, -7, -9, -10],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7-D6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-d6.yaml
    """

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_xlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetELAN_huge(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DarkNetELAN_D(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[80, 64],
        num_blocks=[2, 7, 7, 7, 7, 7],
        id_concat=[-1, -3, -5, -7, -8],
        channel_scale=2,
        final_channel_scale=1,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
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


def DarkNetELAN_huge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv7-E6E
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/WongKinYiu/yolov7/blob/main/cfg/deploy/yolov7-e6e.yaml
    """

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2.merger",
        "stage2.block2.merger",
        "stage3.block2.merger",
        "stage4.block2.merger",
    ]

    return create_model_backbone(
        model_fn=DarkNetELAN_huge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
