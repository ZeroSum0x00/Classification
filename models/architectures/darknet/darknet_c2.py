"""
    DarknetC2: YOLOv8 Backbone with C2f Blocks

    Overview:
        The YOLOv8 backbone is a lightweight and efficient convolutional feature extractor
        built on a modernized architecture using C2f (Concatenate-Conv-Concat-Fuse) blocks.
        It is designed to extract multi-scale semantic features with high computational efficiency,
        enabling better performance across tasks such as object detection and segmentation.

    Key Characteristics:
        - Uses C2f blocks: enhanced feature fusion through multiple bottleneck paths
        - Replaces CSP modules from YOLOv5/YOLOv7 with simpler, more efficient structure
        - Strided convolution is used for downsampling (no pooling layers)
        - Includes SPPF (Spatial Pyramid Pooling - Fast) at the final stage
        - Focuses on modular design: each stage has clearly defined depth and width scaling

    C2f Block:
        - A C2f block splits the input into multiple parts, applies lightweight convs
          (often bottlenecks) to a subset, and fuses everything by concatenation followed
          by a 1x1 conv to reduce dimensionality.
        - Compared to CSP, C2f avoids explicit partial connections and reduces overhead.
        
    General Model Architecture:
         --------------------------------------------------------------------------------
        | Stage                  | Layer                       | Output Shape            |
        |------------------------+-----------------------------+-------------------------|
        | Input                  | input_layer                 | (None, 640, 640, 3)     |
        |------------------------+-----------------------------+-------------------------|
        | Stem                   | ConvolutionBlock (3x3, s=2) | (None, 320, 320, C)     |
        |------------------------+-----------------------------+-------------------------|
        | Stage 1                | ConvolutionBlock (3x3, s=2) | (None, 160, 160, 2C)    |
        |                        | C2f (2x)                    | (None, 160, 160, 2C)    |
        |------------------------+-----------------------------+-------------------------|
        | Stage 2                | ConvolutionBlock (3x3, s=2) | (None, 80, 80, 4C)      |
        |                        | C2f (4x)                    | (None, 80, 80, 4C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 3                | ConvolutionBlock (3x3, s=2) | (None, 40, 40, 8C)      |
        |                        | C2f (4x)                    | (None, 40, 40, 8C)      |
        |------------------------+-----------------------------+-------------------------|
        | Stage 4                | ConvolutionBlock (3x3, s=2) | (None, 20, 20, 16C*S)   |
        |                        | C2f (2x)                    | (None, 20, 20, 16C*S)   |
        |                        | SPPF                        | (None, 20, 20, 16C*S)   |
        |------------------------+-----------------------------+-------------------------|
        | CLS Logics             | GlobalAveragePooling        | (None, 16C*S)           |
        |                        | fc (Logics)                 | (None, 1000)            |
         --------------------------------------------------------------------------------

    Model Parameter Comparison:
         ----------------------------------------
        |      Model Name      |    Params       |
        |----------------------+-----------------|
        |    DarkNetC2 nano    |    1,534,680    |
        |----------------------+-----------------|
        |    DarkNetC2 small   |    5,602,760    |
        |----------------------+-----------------|
        |    DarkNetC2 medium  |   12,449,464    |
        |----------------------+-----------------|
        |    DarkNetC2 large   |   20,344,744    |
        |----------------------+-----------------|
        |    DarkNetC2 xlarge  |   31,613,080    |
         ----------------------------------------

    References:
        - Ultralytics YOLOv8 documentation:
          https://github.com/ultralytics/ultralytics
          
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D,
    Dense, Dropout, GlobalAveragePooling2D,
    concatenate, add,
)

from .darknet19 import ConvolutionBlock
from .darknet_c3 import Bottleneck, C3, SPP, SPPF
from ..vgg.repvgg import RepVGGBlock
from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import (
    process_model_input, create_model_backbone, create_layer_instance,
    validate_conv_arg, check_regularizer,
)



class SimpleRepVGG(tf.keras.layers.Layer):
    
    """
        Simplified RepConv module with Conv fusing
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        groups=1,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = validate_conv_arg(kernel_size)
        self.strides = validate_conv_arg(strides)
        self.padding = padding
        self.dilation_rate = validate_conv_arg(dilation_rate)
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.deploy = deploy

    def build(self, input_shape):
        self.conv1 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.norm  = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        out = x1 + x2
        out = self.norm(out)
        out = self.activ(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class LightConvolutionBlock(tf.keras.layers.Layer):
    
    """
        Light convolution. https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
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
            kernel_size=self.kernel_size,
            strides=(1, 1),
            groups=self.filters,
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
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
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


class ChannelAttention(tf.keras.layers.Layer):
    
    """
        Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """
    
    def __init__(
        self,
        activation="sigmoid",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)

    def build(self, input_shape):
        self.pool = GlobalAveragePooling2D(keepdims=True)

        self.conv = ConvolutionBlock(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
        )

    def call(self, inputs, training=False):
        x = self.pool(inputs)
        x = self.conv(x, training=training)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

class SpatialAttention(tf.keras.layers.Layer):
    
    """
        Spatial-attention module.
    """
    
    def __init__(
        self,
        kernel_size=(3, 3),
        activation="sigmoid",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kernel_size = validate_conv_arg(kernel_size)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)

    def build(self, input_shape):
        self.conv = ConvolutionBlock(
            filters=1,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            activation=self.activation,
            normalizer=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
        )
        
    def call(self, inputs, training=False):
        mean_value = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_value  = tf.reduce_max(inputs, axis=-1, keepdims=True)
        merger = concatenate([mean_value, max_value], axis=-1)
        x = self.conv(merger, training=training)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class CBAM(tf.keras.layers.Layer):
    
    """
        Convolutional Block Attention Module.
    """
    
    def __init__(
        self,
        kernel_size=(3, 3),
        activation="sigmoid",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kernel_size = validate_conv_arg(kernel_size)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=self.kernel_size,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
        )

    def call(self, inputs, training=False):
        x = self.channel_attention(inputs, training=training)
        x = self.spatial_attention(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class Conv2DCustomKernel(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
        dilations=(1, 1),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = validate_conv_arg(kernel_size)
        self.strides = validate_conv_arg(strides)
        self.padding = padding
        self.dilations = validate_conv_arg(dilations)
        
    def build(self, input_shape):
        kernel_init = tf.range(0, self.filters, 1, dtype=tf.float32)
        shape = list(self.kernel_size) + [1, self.filters]
        kernel_init = tf.reshape(kernel_init, shape=shape)
        
        self.kernel = tf.Variable(
            name="kernel_value",
            initial_value=kernel_init,
            dtype=tf.float32,
            trainable=False,
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = tf.nn.conv2d(
            input=inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilations,
        )
        x = tf.stop_gradient(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilations": self.dilations
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class DFL(tf.keras.layers.Layer):
    
    """
        Integral module of Distribution Focal Loss (DFL).
        Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters

    def build(self, input_shape):
        bs = input_shape[-1]
        self.conv = Conv2DCustomKernel(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
        )

    def call(self, inputs, training=False):        
        _, anchor, dim = inputs.shape
        x = tf.reshape(inputs, shape=(-1, anchor, 4, self.filters))
        x = tf.nn.softmax(x, axis=-1)
        x = self.conv(x, training=training)
        x = tf.reshape(x, shape=(-1, 4, anchor))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DFL2(tf.keras.layers.Layer):
    """
    Integral module of Distribution Focal Loss (DFL) in TensorFlow.
    """
    def __init__(self, filters, *args, **kwargs):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__(*args, **kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        """Define the weight initialization."""
        x = tf.range(self.filters, dtype=tf.float32)
        self.conv = Conv2D(1, kernel_size=1, use_bias=False, trainable=False)
        self.conv.kernel.assign(tf.reshape(x, (1, 1, self.filters, 1)))

    def call(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, a, _ = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[1]
        x = tf.reshape(x, (b, 4, self.filters, a))
        x = tf.nn.softmax(x, axis=2)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(self.conv(x), (b, 4, a))

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class Proto(tf.keras.layers.Layer):
    
    """
        YOLOv8 mask Proto module for segmentation models.
    """
    
    def __init__(
        self,
        filters,
        expansion=8,
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
        self.expansion = expansion
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
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.upsample = Conv2DTranspose(
            filters=hidden_dim,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.conv2 = ConvolutionBlock(
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
        x = self.upsample(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
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

        
class HGStem(tf.keras.layers.Layer):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    
    def __init__(
        self,
        filters,
        expansion=8,
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
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        self.conv1 = self.convolution_block(hidden_dim, 3, strides=(2, 2), padding="same")
        self.conv2 = self.convolution_block(hidden_dim // 2, 2, strides=(1, 1))
        self.conv3 = self.convolution_block(hidden_dim, 2, strides=(1, 1))
        self.pool  = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")
        self.conv4 = self.convolution_block(hidden_dim, 3, strides=(2, 2), padding="same")
        self.conv5 = self.convolution_block(self.filters, 1, strides=(1, 1))

    def convolution_block(self, filters, kernel_size, strides=1, padding="valid", groups=1):
        return Sequential([
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=groups,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name(self.activation),
        ])

    def call(self, inputs, training=False):        
        x = self.conv1(inputs, training=training)
        pad_h = (x.shape[1] % 2) + 1
        pad_w = (x.shape[2] % 2) + 1
        pad = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        x = tf.pad(x, pad, mode="CONSTANT", constant_values=0)
        
        x1 = self.conv2(x, training=training)
        
        pad_h = (x1.shape[1] % 2) + 1
        pad_w = (x1.shape[2] % 2) + 1
        pad = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        x1 = tf.pad(x1, pad, mode="CONSTANT", constant_values=0)

        x1 = self.conv3(x1, training=training)
        
        pad_h = (x.shape[1] % 2)
        pad_w = (x.shape[2] % 2)
        pad = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        
        x2 = self.pool(tf.pad(x, pad, mode="CONSTANT", constant_values=0))

        x1_shape = tf.shape(x1)[1:3]
        x2_shape = tf.shape(x2)[1:3]
        
        same_shape = tf.reduce_all(tf.equal(x1_shape, x2_shape))
        
        x2 = tf.cond(
            same_shape,
            lambda: x2,
            lambda: tf.image.resize(x2, size=x1_shape[0:2])
        )

        x = concatenate([x1, x2], axis=-1)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion": self.expansion,
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

        
class HGBlock(tf.keras.layers.Layer):
    
    """
        HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        iters=6,
        expansion=2,
        shortcut=False,
        block=ConvolutionBlock,
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
        self.iters = iters
        self.expansion = expansion
        self.shortcut = shortcut
        self.block = block
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        
        self.module_list = [
            self.block(
                filters=hidden_dim,
                kernel_size=self.kernel_size,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ]
        
        self.sc = ConvolutionBlock(
            filters=self.filters // 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.ec = ConvolutionBlock(
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
        x = [inputs]
        for module in self.module_list:
            x.append(module(x[-1]))

        x = concatenate(x, axis=-1)
        x = self.sc(x, training=training)
        x = self.ec(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "iters": self.iters,
            "expansion": self.expansion,
            "shortcut": self.shortcut,
            "block": self.block,
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

        
class C1(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck with 1 convolution. 
    """
    
    def __init__(
        self,
        filters,
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
        super().__init__(*args, **kwargs)
        self.filters = filters
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
        self.c = input_shape[-1]
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
            ConvolutionBlock(
                filters=self.filters,
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
        ])

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        y = self.middle(x, training=training)
        if self.shortcut:
            y = add([x, y])
        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
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

        
class C2(tf.keras.layers.Layer):
    
    """ 
        CSP Bottleneck with 2 convolution. 
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
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=2 * hidden_dim,
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
        
        self.blocks = Sequential([
            Bottleneck(
                filters=hidden_dim,
                kernels=(3, 3),
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
        x = self.conv1(inputs, training=training)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        x1 = self.blocks(x1, training=training)
        x = concatenate([x1, x2], axis=-1)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
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

        
class C2f(tf.keras.layers.Layer):
    
    """ 
        Faster Implementation of CSP Bottleneck with 2 convolutions. 
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
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = ConvolutionBlock(
            filters=2 * hidden_dim,
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
        
        self.blocks = [
            Bottleneck(
                filters=hidden_dim,
                kernels=(3, 3),
                shortcut=self.shortcut,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
            )
            for _ in range(self.iters)
        ]


    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = tf.split(x, num_or_size_splits=2, axis=-1)
        for block in self.blocks:
            x.append(block(x[-1]))
        x = concatenate(x, axis=-1)
        x = self.conv2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
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

        
class C3Rep(C3):
    
    """ 
        C3 module with Rep-convolutions. 
    """

    def __init__(
        self,
        filters,
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
            RepVGGBlock(
                filters=hidden_dim,
                kernel_size=(3, 3),
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                deploy=self.deploy,
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
        
        if hidden_dim != self.filters:
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
        
        merger = add([x, y])
        if self.shortcut and hasattr(self, "conv2"):
            merger = self.conv2(merger, training=training)
        return merger

    def get_config(self):
        config = super().get_config()
        config.update({
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def DarkNetC2(
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
        x = create_layer_instance(
            extractor_block1,
            filters=filters[0],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    last_stage_idx = len(num_blocks) - 2
    final_filters = None
    for i, num_block in enumerate(num_blocks[1:]):
        is_last_stage = (i == last_stage_idx)
        block_name_prefix = f"stage{i + 1}"

        f = filters[i + 1]
        
        if is_last_stage:
            f = int(f * final_channel_scale)
            final_filters = f
            
        if num_block > 0:
            x = create_layer_instance(
                extractor_block1 if i == 0 else extractor_block2,
                filters=f,
                kernel_size=(3, 3),
                strides=(2, 2),
                **layer_constant_dict,
                name=f"{block_name_prefix}.block1"
            )(x)

        if num_block > 1:
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=f,
                iters=num_block - 1,
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

    model_name = "DarkNet-C2"
    if filters == [16, 32, 64, 128, 256] and num_blocks == [1, 2, 3, 3, 2]:
        model_name += "-nano"
    elif filters == [32, 64, 128, 256, 512] and num_blocks == [1, 2, 3, 3, 2]:
        model_name += "-small"
    elif filters == [48, 96, 192, 384, 768] and num_blocks == [1, 3, 5, 5, 3]:
        model_name += "-medium"
    elif filters == [64, 128, 256, 512, 1024] and num_blocks == [1, 4, 7, 7, 4]:
        model_name += "-large"
    elif filters == [80, 160, 320, 640, 1280] and num_blocks == [1, 4, 7, 7, 4]:
        model_name += "-xlarge"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def DarkNetC2_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    num_blocks,
    filters,
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
        model_fn=DarkNetC2,
        custom_layers=custom_layers,
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetC2_nano(
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
    
    model = DarkNetC2(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C2f,
        pyramid_pooling=SPPF,
        filters=16,
        num_blocks=[1, 2, 3, 3, 2],
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


def DarkNetC2_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv8 version nano
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetC2_nano,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetC2_small(
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
    
    model = DarkNetC2(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C2f,
        pyramid_pooling=SPPF,
        filters=32,
        num_blocks=[1, 2, 3, 3, 2],
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


def DarkNetC2_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv8 version small
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetC2_small,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetC2_medium(
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
    
    model = DarkNetC2(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C2f,
        pyramid_pooling=SPPF,
        filters=48,
        num_blocks=[1, 3, 5, 5, 3],
        channel_scale=2,
        final_channel_scale=0.75,
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


def DarkNetC2_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv8 version medium
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetC2_medium,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetC2_large(
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
    
    model = DarkNetC2(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C2f,
        pyramid_pooling=SPPF,
        filters=64,
        num_blocks=[1, 4, 7, 7, 4],
        channel_scale=2,
        final_channel_scale=0.5,
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


def DarkNetC2_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv8 version large
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetC2_large,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DarkNetC2_xlarge(
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
    
    model = DarkNetC2(
        feature_extractor=ConvolutionBlock,
        fusion_layer=C2f,
        pyramid_pooling=SPPF,
        filters=80,
        num_blocks=[1, 4, 7, 7, 4],
        channel_scale=2,
        final_channel_scale=0.5,
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


def DarkNetC2_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv8 version xlarge
        - In YOLOv8, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
    """

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
    ]

    return create_model_backbone(
        model_fn=DarkNetC2_xlarge,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    