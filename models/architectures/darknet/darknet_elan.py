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
      |    DarkNetELAN small     |   21,582,016    |
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

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Dense, Dropout, MaxPooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D, Concatenate, concatenate, add
)
from tensorflow.keras.regularizers import l2

from .darknet53 import ConvolutionBlock
from .darknet_c3 import Bottleneck, GhostConv, GhostBottleneck
from .efficient_rep import CSPSPPF, RepVGGBlock

from models.layers import get_activation_from_name, get_normalizer_from_name, ScaleWeight, LinearLayer
from utils.model_processing import process_model_input, create_layer_instance



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


class ChunCat(tf.keras.layers.Layer):
    
    def __init__(self, chun_dim=2, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chun_dim = chun_dim
        self.axis     = axis
        self.merger   = Concatenate(axis=self.axis)
        
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


class Shortcut(tf.keras.layers.Layer):
    
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        
    def call(self, inputs, shortcut=None):
        if shortcut is not None:
            return inputs + shortcut
        else:
            return inputs


class FoldCut(tf.keras.layers.Layer):
    
    def __init__(self, fold_dim=2, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_dim = fold_dim
        self.axis     = axis
        
    def call(self, inputs):
        x1, x2 = tf.split(
            inputs,
            num_or_size_splits=[self.fold_dim, inputs.shape[self.axis] - self.fold_dim],
            axis=self.axis
        )
        return x1 + x2


class ImplicitAdd(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, stddev=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean   = mean
        self.stddev = stddev
                     
    def build(self, input_shape):
        init_value = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.stddev)
        self.implicit = tf.Variable(
            name="implicit",
            initial_value=init_value(shape=(1, 1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True)
        
    def call(self, inputs, training=False):
        return inputs + self.implicit


class ImplicitMul(tf.keras.layers.Layer):
    def __init__(self, mean=1.0, stddev=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.conv_scale_init = conv_scale_init
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
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
                    kernel_regularizer=l2(self.regularizer_decay),
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
            kernel_regularizer=l2(self.regularizer_decay),
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
            "normalizer": self.normalizer
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
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
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
            kernel_regularizer=l2(self.regularizer_decay),
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
            "normalizer": self.normalizer
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
        self.pool_size = pool_size
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
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
        self.pool_size = pool_size
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
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
                kernel_regularizer=l2(self.regularizer_decay),
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
        self.regularizer_decay = regularizer_decay
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
        self.regularizer_decay = regularizer_decay
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
        self.regularizer_decay = regularizer_decay
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
    pooling=None,
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

    
    # Stage 0:
    x = create_layer_instance(
        extractor_block1,
        filters=f0,
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stem"
    )(inputs)


    # Stage 1:
    x = create_layer_instance(
        extractor_block1,
        filters=f0 * channel_scale,
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage1"
    )(x)
    

    # Stage 2:
    x = create_layer_instance(
        extractor_block2,
        filters=f0,
        kernel_size=(1, 1),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage2.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block1,
        filters=[f1, f0 * channel_scale],
        iters=num_blocks[0],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage2.block2"
    )(x)

    
    # Stage 3:
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="stage3.block1")(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=f0 * channel_scale,
        kernel_size=(1, 1),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage3.block2"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale, f0 * channel_scale**2],
        iters=num_blocks[1],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage3.block3"
    )(x)


    # Stage 4:
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="stage4.block1")(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=f0 * channel_scale**2,
        kernel_size=(1, 1),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage4.block2"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale**2, f0 * channel_scale**3],
        iters=num_blocks[2],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage4.block3"
    )(x)


    # Stage 5:
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="stage5.block1")(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=int(f0 * channel_scale**3 * final_channel_scale),
        kernel_size=(1, 1),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage5.block2"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale**3, int(f0 * channel_scale**4 * final_channel_scale)],
        iters=num_blocks[3],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage5.block3"
    )(x)

    if pyramid_pooling:
        for i, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(f0 * channel_scale**4 * final_channel_scale),
                **layer_constant_dict,
                name=f"stage5.block{i + 4}"
            )(x)
    else:
        x = LinearLayer(name="stage5.block4")(x)
        
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

    if filters == [32, 32] and num_blocks == [2, 2, 2, 2]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Tiny")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-A")
        
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

    model = DarkNetELAN_A(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage2.block2",
        "stage3.block3",
        "stage4.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


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

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
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

        
    # Stage 0:
    x = create_layer_instance(
        extractor_block1,
        filters=f0,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stem.block1"
    )(inputs)
    
    x = create_layer_instance(
        extractor_block1,
        filters=f0 * channel_scale,
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stem.block2"
    )(x)
    
    x = create_layer_instance(
        extractor_block1,
        filters=f0 * channel_scale,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stem.block3"
    )(x)
    

    # Stage 1:
    x = create_layer_instance(
        extractor_block1,
        filters=f0 * channel_scale**2,
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage1.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block1,
        filters=[f1, f0 * channel_scale**3],
        iters=num_blocks[0],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage1.block2"
    )(x)
    

    # Stage 2:
    x = create_layer_instance(
        extractor_block2,
        filters=f0 * channel_scale**2,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage2.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale, f0 * channel_scale**4],
        iters=num_blocks[1],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage2.block2"
    )(x)
    

    # Stage 3:
    x = create_layer_instance(
        extractor_block2,
        filters=f0 * channel_scale**3,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage3.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale**2, f0 * channel_scale**5],
        iters=num_blocks[2],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage3.block2"
    )(x)

    
    # Stage 4:
    x = create_layer_instance(
        extractor_block2,
        filters=int(f0 * channel_scale**4 * final_channel_scale),
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stage4.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[f1 * channel_scale**2, int(f0 * channel_scale**5 * final_channel_scale)],
        iters=num_blocks[3],
        id_concat=id_concat,
        **layer_constant_dict,
        name="stage4.block2"
    )(x)

    if pyramid_pooling:
        for i, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(f0 * channel_scale**5 * final_channel_scale),
                **layer_constant_dict,
                name=f"stage4.block{i + 3}"
            )(x)
    else:
        x = LinearLayer(name="stage4.block3")(x)
        
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

    if filters == [32, 64] and num_blocks == [4, 4, 4, 4]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Nano")
    elif filters == [40, 64] and num_blocks == [6, 6, 6, 6]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Small")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-B")
        
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

    model = DarkNetELAN_B(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


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
    pooling=None,
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

    # if fusion_layer and fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
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

    scale = [
        i for i in [
            channel_scale,
            channel_scale**2,
            channel_scale**3,
            channel_scale**4 - channel_scale**2,
            int(channel_scale**4 * final_channel_scale),
        ]
    ]
    scale_f0 = [f0 * i for i in scale]
    scale_f1 = [f1 * (i // 2) for i in scale]

    x = ReOrg(name="stem.block1")(inputs)
    
    x = create_layer_instance(
        extractor_block1,
        filters=f0,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stem.block2"
    )(x)

    for i, (s_f0, s_f1) in enumerate(zip(scale_f0, scale_f1)):
        x = create_layer_instance(
            extractor_block1 if i < 1 else extractor_block2,
            filters=s_f0,
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        x = create_layer_instance(
            fusion_block1 if i < 1 else fusion_block2,
            filters=[s_f1, s_f0],
            iters=num_blocks[i],
            id_concat=id_concat,
            **layer_constant_dict,
            name=f"stage{i + 1}.block2"
        )(x)
        
    if pyramid_pooling:
        for j, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=scale_f0[-1],
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

    if filters == [64, 64] and num_blocks == [4, 4, 4, 4, 4]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Medium")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-C")
        
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

    model = DarkNetELAN_C(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


    
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

    # if feature_extractor.__name__ not in ["Focus", "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, ConvolutionBlock, GhostConv].")

    # if fusion_layer.__name__ not in ["C3", "C3x", "C3SPP", "C3SPPF", "C3Ghost", "C3Trans", "BottleneckCSP",
    #                                  "HGBlock", "C1", "C2", "C2f", "C3Rep"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [C3, C3x, C3SPP, C3SPPF, C3Ghost, C3Trans, BottleneckCSP, \
    #                                                                               HGBlock, C1, C2, C2f, C3Rep].")

    # if pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
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


    scale = [
        i for i in [
            channel_scale,
            channel_scale**2,
            channel_scale**3,
            channel_scale**4 - channel_scale**2,
            int(channel_scale**4 * final_channel_scale),
        ]
    ]
    scale_f0 = [f0 * i for i in scale]
    scale_f1 = [f1 * (i // 2) for i in scale]

    x = ReOrg(name="stem.block1")(inputs)
    
    x = ConvolutionBlock(
        filters=f0,
        kernel_size=(3, 3),
        strides=(1, 1),
        **layer_constant_dict,
        name="stem.block2"
    )(x)

    for i, (s_f0, s_f1) in enumerate(zip(scale_f0, scale_f1)):
        x = create_layer_instance(
            extractor_block1 if i < 1 else extractor_block2,
            filters=s_f0,
            pool_size=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        x = create_layer_instance(
            fusion_block1 if i < 1 else fusion_block2,
            filters=[s_f1, s_f0],
            iters=num_blocks[i],
            id_concat=id_concat,
            **layer_constant_dict,
            name=f"stage{i + 1}.block2"
        )(x)

    if pyramid_pooling:
        for j, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=scale_f0[-1],
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

    if filters == [80, 64] and num_blocks == [6, 6, 6, 6, 6]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Large")
    elif filters == [96, 64] and num_blocks == [8, 8, 8, 8, 8]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-XLarge")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-D")
        
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

    model = DarkNetELAN_D(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
        "stage5.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_E(
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
    pooling=None,
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

    scale = [
        i for i in [
            channel_scale,
            channel_scale**2,
            channel_scale**3,
            channel_scale**4 - channel_scale**2,
            int(channel_scale**4 * final_channel_scale),
        ]
    ]
    scale_f0 = [f0 * i for i in scale]
    scale_f1 = [f1 * (i // 2) for i in scale]

    x = ReOrg(name="stem.block1")(inputs)
    
    x = ConvolutionBlock(
        filters=f0,
        kernel_size=(3, 3),
        **layer_constant_dict,
        name="stem.block2"
    )(x)

    for i, (s_f0, s_f1) in enumerate(zip(scale_f0, scale_f1)):
        x = create_layer_instance(
            extractor_block1 if i < 1 else extractor_block2,
            filters=s_f0,
            pool_size=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        x1 = create_layer_instance(
            fusion_block1 if i < 1 else fusion_block2,
            filters=[s_f1, s_f0],
            iters=num_blocks[i],
            id_concat=id_concat,
            **layer_constant_dict,
            name=f"stage{i + 1}.block2"
        )(x)
        
        x2 = create_layer_instance(
            fusion_block1 if i < 1 else fusion_block2,
            filters=[s_f1, s_f0],
            iters=num_blocks[i],
            id_concat=id_concat,
            **layer_constant_dict,
            name=f"stage{i + 1}.block3"
        )(x1)
        
        x = Shortcut(name=f"stage{i + 1}.block4")(x1, x2)

    if pyramid_pooling:
        for j, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=scale_f0[-1],
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + 6}"
            )(x)
    else:
        x = LinearLayer(name=f"stage{i + 1}.block5")(x)
        
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

    if filters == [80, 64] and num_blocks == [6, 6, 6, 6, 6]:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-Huge")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN-E")
        
    return model


def DarkNetELAN_E_backbone(
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

    model = DarkNetELAN_E(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        id_concat=id_concat,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )
    
    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block4",
        "stage2.block4",
        "stage3.block4",
        "stage4.block4",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_tiny(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
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
        num_blocks=[2, 2, 2, 2],
        id_concat=[-1, -2, -3, -4],
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
    
    model = DarkNetELAN_tiny(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2",
        "stage3.block3",
        "stage4.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_nano(
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
) -> Model:
    
    model = DarkNetELAN_B(
        feature_extractor=[ConvolutionBlock, TransitionBlock],
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[32, 64],
        num_blocks=[4, 4, 4, 4],
        id_concat=[-1, -3, -5, -6],
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
    
    model = DarkNetELAN_nano(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_small(
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
) -> Model:
    
    model = DarkNetELAN_B(
        feature_extractor=[ConvolutionBlock, TransitionBlock],
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[40, 64],
        num_blocks=[6, 6, 6, 6],
        id_concat=[-1, -3, -5, -7, -8],
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
    
    model = DarkNetELAN_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_medium(
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
) -> Model:
    
    model = DarkNetELAN_C(
        feature_extractor=ConvolutionBlock,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[64, 64],
        num_blocks=[4, 4, 4, 4, 4],
        id_concat=[-1, -3, -5, -6],
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
    
    model = DarkNetELAN_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_large(
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
) -> Model:
    
    model = DarkNetELAN_D(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[80, 64],
        num_blocks=[6, 6, 6, 6, 6],
        id_concat=[-1, -3, -5, -7, -8],
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
    
    model = DarkNetELAN_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_xlarge(
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
) -> Model:
    
    model = DarkNetELAN_D(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[96, 64],
        num_blocks=[8, 8, 8, 8, 8],
        id_concat=[-1, -3, -5, -7, -9, -10],
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
    
    model = DarkNetELAN_xlarge(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN_huge(
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
) -> Model:
    
    model = DarkNetELAN_E(
        feature_extractor=DownC,
        fusion_layer=ScaleUpConcatBlock,
        pyramid_pooling=None,
        filters=[80, 64],
        num_blocks=[6, 6, 6, 6, 6],
        id_concat=[-1, -3, -5, -7, -8],
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
    
    model = DarkNetELAN_huge(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block2",
        "stage1.block4",
        "stage2.block4",
        "stage3.block4",
        "stage4.block4",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
