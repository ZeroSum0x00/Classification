"""
  # Description:
    - The following table comparing the params of the DarkNet ELAN4 (YOLOv9 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       --------------------------------------------
      |      Model Name          |    Params       |
      |--------------------------------------------|
      |    DarkNetELAN small     |    8,863,400    |
      |--------------------------------------------|
      |    DarkNetELAN base      |   12,697,256    |
      |--------------------------------------------|
      |    DarkNetELAN large     |   56,904,104    |
      |--------------------------------------------|
      |    DarkNetELAN xlarge    |   74,795,432    |
       --------------------------------------------

  # Reference:
    - Source: https://github.com/WongKinYiu/yolov9

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate, add
)
from tensorflow.keras.regularizers import l2

from .darknet53 import ConvolutionBlock
from .darknet_c3 import Bottleneck, BottleneckCSP
from .darknet_elan import BottleneckCSPA

from models.layers import get_activation_from_name, LinearLayer
from utils.model_processing import process_model_input, create_layer_instance


class AverageConvolutionBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size=3,
        activation="leaky",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super(AverageConvolutionBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
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
            strides=(2, 2),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="SAME")
        
    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv(x, training=training)
        return x


class AverageConvolutionDown(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size=3,
        activation="leaky",
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
            kernel_size=self.kernel_size,
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

    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=1,
        groups=1,
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
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            groups=self.groups,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.conv2 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=self.strides,
            groups=self.groups,
            activation=None,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.activ = get_activation_from_name(self.activation)
        
    def call(self, inputs, training=False):
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)        
        x = x1 + x2
        x = self.activ(x, training=training)
        return x


class RepBottleneck(Bottleneck):
    
    def __init__(
        self,
        filters,
        kernels=(3, 3),
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
            *args, **kwargs,
        )
        self.kernels = kernels
        
    def build(self, input_shape):
        self.c     = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv1 = RepSimpleBlock(
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
        

class ResNBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
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
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        
        if self.shortcut and self.c == self.filters:
            x = add([inputs, x])
        return x


class RepResNBlock(ResNBlock):
    
    def __init__(
        self,
        filters,
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
            *args, **kwargs,
        )
        
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
        
        self.conv2 = RepSimpleBlock(
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
            downsample=self.downsample,
            groups=self.groups,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        

class BottleneckCSPA2(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            Bottleneck(
                filters=hidden_dim,
                kernels=(3, 3),
                strides=(1, 1),
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


class BottleneckCSP2(BottleneckCSP):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.middle = Sequential([
            Bottleneck(
                filters=hidden_dim,
                kernels=(3, 3),
                strides=(1, 1),
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


class RepNCSP(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            RepBottleneck(
                filters=hidden_dim,
                kernels=(3, 3),
                strides=(1, 1),
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


class BaseBottleneckCSP(BottleneckCSPA):

    def build(self, input_shape):
        super().build(input_shape)
        hidden_dim = int(self.filters * self.expansion)
        
        self.block = Sequential([
            Bottleneck(
                filters=hidden_dim,
                kernels=(1, 3),
                strides=(1, 1),
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


class ASPP(tf.keras.layers.Layer):
    
    def __init__(
        self,
        filters,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.kernel_list = [1, 3, 3, 1]
        self.dilation_list = [1, 3, 6, 1]
        self.padding_list = [0, 1, 1, 0]

    def build(self, input_shape):
        self.gap   = GlobalAveragePooling2D(keepdims=True)
        
        self.block = [
            Conv2D(
                filters=self.filters,
                kernel_size=k,
                strides=(1, 1),
                padding="SAME" if p else "VALID",
                dilation_rate=d,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=l2(self.regularizer_decay),
            )
            for k, p, d in zip(self.kernel_list, self.padding_list, self.dilation_list)
        ]

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

    def __init__(
        self,
        filters,
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
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        if isinstance(self.filters, (tuple, list)):
            f0, f1 = self.filters
        else:
            f0 = f1 = self.filters
            
        self.conv1 = ConvolutionBlock(
            filters=f0,
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
            filters=f1,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.block = [
            MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME"),
            MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME"),
            MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="SAME"),
        ]
        
    def call(self, inputs, training=False):
        y = self.conv1(inputs, training=training)
        out = [y]
        for bk in self.block:
            out.append(bk(out[-1], training=training))

        x = concatenate(out, axis=-1)
        x = self.conv2(x, training=training)
        return x


class RepNCSPELAN4(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        iters=1,
        groups=1,
        expansion=0.5,
        shortcut=True,
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
        self.iters = iters
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
        if isinstance(self.filters, (tuple, list)):
            f0, f1, f2 = self.filters
        else:
            f0 = f1, f2 = self.filters
            
        self.conv1 = ConvolutionBlock(
            filters=f0,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.block = [
            Sequential([
                RepNCSP(
                    filters=f1,
                    groups=self.groups,
                    iters=self.iters,
                    expansion=self.expansion,
                    shortcut=self.shortcut,
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                ),
                ConvolutionBlock(
                    filters=f1,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                )
            ]),
            Sequential([
                RepNCSP(
                    filters=f1,
                    groups=self.groups,
                    iters=self.iters,
                    expansion=self.expansion,
                    shortcut=self.shortcut,
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                ),
                ConvolutionBlock(
                    filters=f1,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                )
            ])
        ]
        
        self.conv2 = ConvolutionBlock(
            filters=f2,
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
        out = self.conv1(inputs, training=training)
        out = tf.split(out, num_or_size_splits=2, axis=-1)
        
        for bk in self.block:
            out.append(bk(out[-1], training=training))

        x = concatenate(out, axis=-1)
        x = self.conv2(x, training=training)
        return x


class CBLinear(tf.keras.layers.Layer):

    def __init__(
        self,
        split_filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        groups=1,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.split_filters = split_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        
    def build(self, input_shape):
        self.conv1 = Conv2D(
            filters=sum(self.split_filters),
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=self.groups,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = tf.split(x, num_or_size_splits=self.split_filters, axis=-1)
        return x


class CBFuse(tf.keras.layers.Layer):

    def __init__(self, fuse_index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_index = fuse_index
                    
    def build(self, input_shape):
        self.target_size = input_shape[0][1:-1]

    def call(self, inputs, training=False):
        res = []
        for idx, feature in enumerate(inputs[1:]):
            x = tf.image.resize(
                feature[self.fuse_index[idx]],
                size=self.target_size,
                method=tf.image.ResizeMethod.BILINEAR,
            )
            res.append(x)
            
        res.append(inputs[0])
        out = tf.stack(res, axis=0)
        out = tf.reduce_sum(out, axis=0)
        return out

        
def DarkNetELAN4_A(
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
    drop_rate=0.1,
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

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks) - 1)]
         
    x = inputs
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            extractor_block1,
            filters=filters[0],
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    for i in range(len(num_blocks) - 1):
        if i == len(num_blocks) - 2:
            f1 = filters[i]
            f2 = [filters[i], filters[i - 1], filters[i]]
        elif i == len(num_blocks) - 3:            
            f1 = filters[i + 1]
            f2 = [filters[i + 1], filters[i], filters[i + 1]]
        else:            
            f1 = filters[i + 1]
            f2 = [filters[i + 1], filters[i], filters[i + 2]]
        
        x = create_layer_instance(
            extractor_block1 if i == 0 else extractor_block2,
            filters=int(f1 * final_channel_scale) if i == len(num_blocks) - 2 else f1,
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        x = create_layer_instance(
            fusion_block1 if i == 0 else fusion_block2,
            filters=[int(f * final_channel_scale) for f in f2] if i == len(num_blocks) - 2 else f2,
            iters=num_blocks[i + 1],
            shortcut=True,
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

    if filters == [64, 128, 256, 512] and num_blocks == [1, 1, 1, 1, 1]:
        if extractor_block2 == AverageConvolutionDown:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-Small")
        elif extractor_block2 == ConvolutionBlock:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-Base")
        else:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-A")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-A")

    return model


def DarkNetELAN4_A_backbone(
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

    model = DarkNetELAN4_A(
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


def DarkNetELAN4_B(
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

    filters = [filters * channel_scale ** i for i in range(5)]
    
    # Stage 0:
    x = create_layer_instance(
        extractor_block1,
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stem.branch1.block1"
    )(inputs)
    
    y0 = create_layer_instance(
        extractor_block1,
        filters=filters[0],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stem.branch2.block1"
    )(inputs)


    # Stage 1:
    x = create_layer_instance(
        extractor_block1,
        filters=filters[1],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage1.branch1.block1"
    )(x)
    
    y1 = CBLinear([filters[0]], name="stage1.branch2.block1")(x)

    x = create_layer_instance(
        fusion_block1,
        filters=[filters[1], filters[0], filters[2]],
        iters=num_blocks[0],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage1.branch1.block2"
    )(x)
    
    y2 = CBLinear([filters[0], filters[1]], name="stage1.branch2.block2")(x)

    
    # Stage 2:
    x = create_layer_instance(
        extractor_block2,
        filters=filters[2],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage2.branch1.block1"
    )(x)
    
    x = create_layer_instance(
        fusion_block2,
        filters=[filters[2], filters[1], filters[3]],
        iters=num_blocks[1],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage2.branch1.block2"
    )(x)
    
    y3 = CBLinear([filters[0], filters[1], filters[2]], name="stage2.branch2.block1")(x)


    # Stage 3:
    x = create_layer_instance(
        extractor_block2,
        filters=filters[3],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage3.branch1.block1"
    )(x)

    x = create_layer_instance(
        fusion_block2,
        filters=[filters[3], filters[2], filters[4]],
        iters=num_blocks[2],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage3.branch1.block2"
    )(x)
    
    y4 = CBLinear([filters[0], filters[1], filters[2], filters[3]], name="stage3.branch2.block1")(x)


    # Stage 4:
    x = create_layer_instance(
        extractor_block2,
        filters=filters[4],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage4.block1.branch1"
    )(x)

    x = create_layer_instance(
        fusion_block2,
        filters=[filters[3], filters[2], filters[4]],
        iters=num_blocks[3],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage4.branch1.block2"
    )(x)
    
    y5 = CBLinear([filters[0], filters[1], filters[2], filters[3], filters[4]], name="stage4.branch2.block1")(x)


    # Stage 5:
    x = CBFuse(fuse_index=[0, 0, 0, 0, 0], name="stage5.block1")([y0, y1, y2, y3, y4, y5])

    x = create_layer_instance(
        extractor_block1,
        filters=filters[1],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage6.block1",
    )(x)
    
    x = CBFuse(fuse_index=[1, 1, 1, 1, 1], name="stage6.block2")([x, y2, y3, y4, y5])

    x = create_layer_instance(
        fusion_block2,
        filters=[filters[1], filters[0], filters[2]],
        iters=num_blocks[4],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage6.block3"
    )(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=filters[2],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage7.block1",
    )(x)
    
    x = CBFuse(fuse_index=[2, 2, 2], name="stage7.block2")([x, y3, y4, y5])

    x = create_layer_instance(
        fusion_block2,
        filters=[filters[2], filters[1], filters[3]],
        iters=num_blocks[5],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage7.block3"
    )(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=filters[3],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage8.block1",
    )(x)
    
    x = CBFuse(fuse_index=[3, 3], name="stage8.block2")([x, y4, y5])

    x = create_layer_instance(
        fusion_block2,
        filters=[filters[3], filters[2], filters[4]],
        iters=num_blocks[6],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage8.block3"
    )(x)
    
    x = create_layer_instance(
        extractor_block2,
        filters=filters[4],
        kernel_size=(3, 3),
        strides=(2, 2),
        **layer_constant_dict,
        name="stage9.block1",
    )(x)
    
    x = CBFuse(fuse_index=[4], name="stage9.block2")([x, y5])
    
    x = create_layer_instance(
        fusion_block2,
        filters=[filters[3], filters[2], filters[4]],
        iters=num_blocks[7],
        groups=1,
        expansion=0.5,
        shortcut=True,
        **layer_constant_dict,
        name="stage9.block3"
    )(x)
    
    if pyramid_pooling:
        for i, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters * channel_scale**5 * final_channel_scale),
                **layer_constant_dict,
                name=f"stage9.block{i + 4}"
            )(x)
    else:
        x = LinearLayer(name="stage9.block4")(x)
        
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

    if filters == [64, 128, 256, 512, 1024] and num_blocks == [2, 2, 2, 2, 2, 2, 2, 2]:
        if extractor_block2 == AverageConvolutionDown:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-Large")
        elif extractor_block2 == ConvolutionBlock:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-XLarge")
        else:
            model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-B")
    else:
        model = Model(inputs=inputs, outputs=x, name="DarkNet-ELAN4-B")

    return model


def DarkNetELAN4_B_backbone(
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

    model = DarkNetELAN4_B(
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
        "stage5.block1",
        "stage6.block3",
        "stage7.block3",
        "stage8.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN4_small(
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
    
    model = DarkNetELAN4_A(
        feature_extractor=[ConvolutionBlock, AverageConvolutionDown],
        fusion_layer=RepNCSPELAN4,
        pyramid_pooling=None,
        filters=64,
        num_blocks=[1, 1, 1, 1, 1],
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


def DarkNetELAN4_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv9-C
        - In YOLOv9, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov9/blob/main/models/detect/yolov9-c.yaml
    """
    
    model = DarkNetELAN4_small(
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

        
def DarkNetELAN4_base(
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
    
    model = DarkNetELAN4_A(
        feature_extractor=[ConvolutionBlock, ConvolutionBlock],
        fusion_layer=RepNCSPELAN4,
        pyramid_pooling=None,
        filters=64,
        num_blocks=[1, 1, 1, 1, 1],
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


def DarkNetELAN4_base_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv9
        - In YOLOv9, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov9/blob/main/models/detect/yolov9.yaml
    """
    
    model = DarkNetELAN4_base(
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


def DarkNetELAN4_large(
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
    
    model = DarkNetELAN4_B(
        feature_extractor=[ConvolutionBlock, AverageConvolutionDown],
        fusion_layer=RepNCSPELAN4,
        pyramid_pooling=None,
        filters=64,
        num_blocks=[2, 2, 2, 2, 2, 2, 2, 2],
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


def DarkNetELAN4_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv9
        - In YOLOv9, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/WongKinYiu/yolov9/blob/main/models/detect/yolov9.yaml
    """
    
    model = DarkNetELAN4_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage5.block1",
        "stage6.block3",
        "stage7.block3",
        "stage8.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DarkNetELAN4_xlarge(
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
    
    model = DarkNetELAN4_B(
        feature_extractor=[ConvolutionBlock, ConvolutionBlock],
        fusion_layer=RepNCSPELAN4,
        pyramid_pooling=None,
        filters=64,
        num_blocks=[2, 2, 2, 2, 2, 2, 2, 2],
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


def DarkNetELAN4_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = DarkNetELAN4_xlarge(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage5.block1",
        "stage6.block3",
        "stage7.block3",
        "stage8.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
