"""
  # Description:
    - The following table comparing the params of the EfficientRep architectures (YOLOv6 backbone) in Tensorflow on 
    image size 640 x 640 x 3:

       -------------------------------------------------------------------------------
      |         Model Name             |    Un-deploy params    |    Deploy params    |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep lite nano     |           345,070      |        345,070      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep lite medium   |           676,605      |        676,605      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep lite large    |         1,066,285      |      1,066,285      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep nano          |         3,705,928      |      3,393,480      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep6 nano         |         4,858,952      |      4,426,248      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep small         |        14,253,224      |     13,045,672      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep6 small        |        18,853,032      |     17,175,592      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep medium        |        26,534,536      |     24,251,732      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep6 medium       |        37,032,328      |     33,770,134      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep large         |        42,581,928      |     39,531,453      |
      |-------------------------------------------------------------------------------|
      |    Efficient-Rep6 large        |        58,544,040      |     54,590,400      |
      |-------------------------------------------------------------------------------|
      |    Efficient-MBLA small        |         7,656,776      |      7,252,048      |
      |-------------------------------------------------------------------------------|
      |    Efficient-MBLA medium       |        16,806,904      |     15,911,808      |
      |-------------------------------------------------------------------------------|
      |    Efficient-MBLA large        |        29,505,192      |     27,927,728      |
      |-------------------------------------------------------------------------------|
      |    Efficient-MBLA xlarge       |        55,559,080      |     51,591,610      |
       -------------------------------------------------------------------------------

  # Reference:
    - Source: https://github.com/meituan/YOLOv6/tree/main

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Dense, Dropout, MaxPooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
    concatenate
)
from tensorflow.keras.regularizers import l2

from .darknet53 import ConvolutionBlock
from .darknet_c3 import SPPF
from ..vgg.repvgg import SEBlock, RepVGGBlock
from models.layers import (
    ChannelShuffle, ScaleWeight, LinearLayer,
    get_activation_from_name, get_normalizer_from_name
)
from utils.model_processing import process_model_input, create_layer_instance



class CustomLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        expansion=0.5,
        rep_block=RepVGGBlock,
        sub_block=None,
        scale_weight=False,
        iters=1,
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
        self.expansion = expansion
        self.rep_block = rep_block
        self.sub_block = sub_block 
        self.scale_weight = scale_weight
        self.iters = iters
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        self.deploy = deploy


class CSPSPPF(CustomLayer):
    
    """ 
        CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    
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
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.pool_size = pool_size
        
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
        
        self.conv5 = ConvolutionBlock(
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
        
        self.conv6 = ConvolutionBlock(
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
        
        self.shortcut = ConvolutionBlock(
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
        
        self.pool = MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding="same")

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
    
    """
        A CSLA block is a LinearAddBlock with is_csla=True
    """
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        dilation=1,
        groups=1,
        is_csla=False,
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
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.is_csla = is_csla
        self.conv_scale_init = conv_scale_init
        
    def build(self, input_shape):
        self.conv  = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.scale_layer = ScaleWeight(self.conv_scale_init, use_bias=False)
        
        self.conv_1x1 = Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=self.strides,
            padding="valid",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.scale_1x1 = ScaleWeight(self.conv_scale_init, use_bias=False)
        
        if input_shape[-1] == self.filters and self.strides == (1, 1):
            self.scale_identity = ScaleWeight(1.0, use_bias=False)
            
        self.norm = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activ = get_activation_from_name(self.activation)
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.scale_layer(x, training=False if self.is_csla else training)
        y = self.conv_1x1(inputs, training=training)
        y = self.scale_1x1(y, training=False if self.is_csla else training)
        out = x + y
        if hasattr(self, "scale_identity"):
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
    
    def __init__(
        self,
        filters,
        rep_block=RepVGGBlock,
        scale_weight=False,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super().__init__(
            rep_block=rep_block,
            scale_weight=scale_weight,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            deploy=deploy,
            *args, **kwargs
        )
        self.filters = filters

    def build(self, input_shape):
        if self.rep_block != RepVGGBlock:
            self.conv1 = self.rep_block(
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
            
            self.conv2 = self.rep_block(
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
        else:
            self.conv1 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
            self.conv2 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
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
            "deploy": self.deploy
        })
        return config


class BottleRep3(CustomLayer):
    
    def __init__(
        self,
        filters,
        rep_block=RepVGGBlock,
        scale_weight=False,
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
        self.rep_block = rep_block
        self.scale_weight = scale_weight
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        self.deploy = deploy

    def build(self, input_shape):
        if self.rep_block != RepVGGBlock:
            self.conv1 = self.rep_block(
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
            
            self.conv2 = self.rep_block(
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
            
            self.conv3 = self.rep_block(
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
            
        else:
            self.conv1 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
            self.conv2 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
            self.conv3 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
        if input_shape[-1] != self.filters:
            self.shortcut = False
        else:
            self.shortcut = True
            
        if self.scale_weight:
            with tf.init_scope():
                self.alpha = tf.Variable(name="bottleRep.alpha",
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
            "deploy": self.deploy
        })
        return config


class RepBlock(CustomLayer):
    
    def __init__(
        self,
        filters,
        rep_block=RepVGGBlock,
        iters=1,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super().__init__(
            rep_block=rep_block,
            iters=iters,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            deploy=deploy,
            *args, **kwargs
        )
        self.filters = filters
        self.sub_block = kwargs.get("sub_block", RepVGGBlock)

    def build(self, input_shape):
        if self.rep_block != BottleRep:
            self.conv1 = self.rep_block(
                filters=self.filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
            self.block = Sequential([
                self.rep_block(
                    filters=self.filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                    deploy=self.deploy,
                )
                for _ in range(self.iters - 1)
            ]) if self.iters > 1 else None
        else:
            self.iters = self.iters // 2
            
            self.conv1 = self.rep_block(
                filters=self.filters,
                rep_block=self.sub_block,
                scale_weight=True,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
            self.block = Sequential([
                self.rep_block(
                    filters=self.filters,
                    rep_block=self.sub_block,
                    scale_weight=True,
                    activation=self.activation,
                    normalizer=self.normalizer,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                    deploy=self.deploy,
                )
                for _ in range(self.iters - 1)
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
            "deploy": self.deploy,
        })
        return config


class BepC3(CustomLayer):
    
    def __init__(
        self,
        filters,
        rep_block=BottleRep,
        sub_block=None,
        expansion=0.5,
        iters=1,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super().__init__(
            expansion=expansion,
            rep_block=rep_block,
            sub_block=sub_block or RepVGGBlock,
            iters=iters,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            deploy=deploy,
            *args, **kwargs
        )
        self.filters = filters
        
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
        
        if self.sub_block:
            self.block = RepBlock(
                filters=hidden_dim,
                rep_block=self.rep_block,
                sub_block=self.sub_block,
                iters=self.iters,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
        else:
            self.block = RepBlock(
                filters=hidden_dim,
                rep_block=self.rep_block,
                iters=self.iters,
                activation=self.activation,
                normalizer=self.normalizer,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                regularizer_decay=self.regularizer_decay,
                norm_eps=self.norm_eps,
                deploy=self.deploy,
            )
            
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
            "deploy": self.deploy,
        })
        return config


class MBLABlock(CustomLayer):
    
    def __init__(
        self,
        filters,
        rep_block=BottleRep3,
        sub_block=None,
        expansion=0.5,
        iters=1,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        
        iters = iters // 2
        if iters <= 0:
            iters = 1
            
        super().__init__(
            expansion=expansion,
            rep_block=rep_block,
            sub_block=sub_block or RepVGGBlock,
            iters=iters,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            deploy=deploy,
            *args, **kwargs
        )
        self.filters = filters
        
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

        self.conv1 = ConvolutionBlock(
            filters=self.branch_num * hidden_dim,
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
        
        self.block = []
        for n_list_i in n_list[1:]:
            self.block.append([
                self.rep_block(
                    filters=hidden_dim,
                    rep_block=self.sub_block,
                    scale_weight=True,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    regularizer_decay=self.regularizer_decay,
                    norm_eps=self.norm_eps,
                    deploy=self.deploy,
                )
                for _ in range(n_list_i)
            ])
            
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
            "deploy": self.deploy,
        })
        return config


class BiFusion(CustomLayer):
    
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
        super().__init__(
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        
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
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.upsample = Conv2DTranspose(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.downsample = ConvolutionBlock(
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


class Lite_EffiBlockS1(CustomLayer):
    
    def __init__(
        self,
        filters,
        strides=(1, 1),
        expansion=1,
        activation="hard-swish",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        hidden_dim = int(self.filters * self.expansion)
        
        self.conv_pw = ConvolutionBlock(
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
        
        self.conv_dw = self.convolution_block(
            filters=hidden_dim,
            kernel_size=(3, 3),
            strides=self.strides,
        )
        
        self.conv = ConvolutionBlock(
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
        
        self.se_block  = SEBlock(
            expansion=0.25,
            activation=self.activation,
            normalizer=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.shuffle = ChannelShuffle(2)
        
        super().build(input_shape)

    def convolution_block(self, filters, kernel_size, strides):
        return  Sequential([
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding="same",
                       groups=filters,
                       use_bias=False,
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       kernel_regularizer=l2(self.regularizer_decay),
                      ),
                get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
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


class Lite_EffiBlockS2(Lite_EffiBlockS1):
    def __init__(
        self,
        filters,
        strides=(1, 1),
        expansion=1,
        activation="hard-swish",
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
        channel_dim = input_shape[-1]
        hidden_dim = int(self.filters * self.expansion)

        self.conv_dw_1 = self.convolution_block(
            filters=channel_dim,
            kernel_size=(3, 3),
            strides=self.strides,
        )
        
        self.conv_1 = ConvolutionBlock(
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

        self.conv_pw_2 = ConvolutionBlock(
            filters=hidden_dim // 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        self.conv_dw_2 = self.convolution_block(
            filters=hidden_dim // 2,
            kernel_size=(3, 3),
            strides=self.strides,
        )

        self.se_block = SEBlock(
            expansion=0.25,
            activation=self.activation,
            normalizer=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        self.conv_2 = ConvolutionBlock(
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

        self.conv_dw_3 = ConvolutionBlock(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            groups=self.filters,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )

        self.conv_pw_3 = ConvolutionBlock(
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

        super().build(input_shape)

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
        })
        return config



class DPBlock(CustomLayer):
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        fuse=False,
        activation="hard-swish",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.fuse = fuse
        
    def build(self, input_shape):
        group = self.filters if input_shape[-1] % self.filters == 0 else 1
        
        self.conv_dw_1 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=group,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.norm_1 = get_normalizer_from_name(self.normalizer)
        self.activ_1 = get_activation_from_name(self.activation)
        
        self.conv_pw_1 = Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=l2(self.regularizer_decay),
        )
        
        self.norm_2 = get_normalizer_from_name(self.normalizer)
        self.activ_2 = get_activation_from_name(self.activation)
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
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        expansion=0.5,
        fuse=False,
        activation="hard-swish",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.fuse = fuse
        
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
        
        self.conv2 = DPBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="same",
            fuse=self.fuse,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
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
    
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        expansion=0.5,
        fuse=False,
        activation="hard-swish",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super().__init__(
            expansion=expansion,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            *args, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.fuse = fuse
        
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
        
        self.block = DarknetBlock(
            filters=hidden_dim,
            kernel_size=self.kernel_size,
            expansion=1.0,
            fuse=self.fuse,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
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


def EfficientLite(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    csp_scale=2,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="hard-swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "Conv2D" "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, Conv2D, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
        "deploy": deploy,
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

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks) + 1)]

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

    for i, j in enumerate(num_blocks[1:]):
        x = create_layer_instance(
            extractor_block2,
            filters=filters[i + 1],
            kernel_size=(3, 3),
            strides=(2, 2),
            expansion=csp_scale,
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        for k in range(j - 1):
            x = create_layer_instance(
                fusion_block1 if i == 0 else fusion_block2,
                filters=filters[i + 1],
                kernel_size=(3, 3),
                strides=(1, 1),
                expansion=csp_scale,
                **layer_constant_dict,
                name=f"stage{i + 1}.block{k + 2}"
            )(x)
            
    if pyramid_pooling:
        for l, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters[-1] * final_channel_scale),
                **layer_constant_dict,
                name=f"stage{i + 1}.block{k + l + 3}"
            )(x)
    else:
        x = LinearLayer(name=f"stage{i + 1}.block{k + 3}")(x)

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

    if filters == [24, 32, 48, 96, 176]:
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-Lite-Small")
    elif filters == [24, 32, 64, 144, 288]:
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-Lite-Medium")
    elif filters == [24, 48, 96, 192, 384]:
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-Lite-Large")
    else:
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-Lite")

    return model


def EfficientLite_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    csp_scale=2,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="hard-swish",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = EfficientLite(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        csp_scale=csp_scale,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1" if i == 0 else f"stage{i}.block{j}"
        for i, j in enumerate(num_blocks[:-1])
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientRep(
    feature_extractor=RepVGGBlock,
    fusion_layer=RepBlock,
    pyramid_pooling=CSPSPPF,
    filters=[64, 128, 256, 512, 1024],
    num_blocks=[6, 12, 18, 6],
    csp_scale=1.0,
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
    deploy=False
):

    """
        EfficientRep Backbone
        EfficientRep is handcrafted by hardware-aware neural network design.
        With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """
    
    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    # if feature_extractor and feature_extractor.__name__ not in ["Focus", "Conv2D" "ConvolutionBlock", "GhostConv"]:
    #     raise ValueError(f"Invalid feature_extractor: {feature_extractor}. Expected one of [Focus, Conv2D, ConvolutionBlock, GhostConv].")

    # if fusion_layer and fusion_layer.__name__ not in ["ResidualBlock"]:
    #     raise ValueError(f"Invalid fusion_layer: {fusion_layer}. Expected one of [ResidualBlock].")

    # if pyramid_pooling and pyramid_pooling.__name__ not in ["SPP", "SPPF"]:
    #     raise ValueError(f"Invalid pyramid_pooling: {pyramid_pooling}. Expected one of [SPP, SPPF].")

    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
        "deploy": deploy,
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
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stem.block{i + 1}"
        )(x)

    for i in range(len(num_blocks) - 1):
        x = create_layer_instance(
            extractor_block2,
            filters=filters[i + 1],
            kernel_size=(3, 3),
            strides=(2, 2),
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )(x)
        
        x = create_layer_instance(
            fusion_block1 if i == 0 else fusion_block2,
            filters=filters[i + 1],
            iters=num_blocks[i + 1],
            expansion=csp_scale,
            **layer_constant_dict,
            name=f"stage{i + 1}.block2"
        )(x)

    if pyramid_pooling:
        for k, pooling in enumerate(pyramid_pooling):
            x = create_layer_instance(
                pooling,
                filters=int(filters[-1] * final_channel_scale),
                **layer_constant_dict,
                name=f"stage{i + 1}.block{k + 3}"
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

    if fusion_block1 == MBLABlock and fusion_block2 == MBLABlock:
        name = "MLBA"
    elif fusion_block1 == RepBlock and fusion_block2 == RepBlock:
        name = "Rep"
    else:
        name = "Mishmash"
        
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
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-{name}{len(filters)}-{suffit}")
    else:
        model = Model(inputs=inputs, outputs=x, name=f"Efficient-{name}-{suffit}")

    return model


def EfficientRep_backbone(
    feature_extractor,
    fusion_layer,
    pyramid_pooling,
    filters,
    num_blocks,
    csp_scale=1.0,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = EfficientRep(
        feature_extractor=feature_extractor,
        fusion_layer=fusion_layer,
        pyramid_pooling=pyramid_pooling,
        filters=filters,
        num_blocks=num_blocks,
        csp_scale=csp_scale,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1" if i == 0 else f"stage{i}.block2"
        for i, j in enumerate(num_blocks[:-1])
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientLite_small(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="hard-swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:
    
    model = EfficientLite(
        feature_extractor=[ConvolutionBlock, Lite_EffiBlockS2],
        fusion_layer=Lite_EffiBlockS1,
        pyramid_pooling=None,
        filters=[24, 32, 48, 96, 176],
        num_blocks=[1, 1, 3, 7, 3],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientLite_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv6 lite version small
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_s.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_s_finetune.py
    """

    model = EfficientLite_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1",
        "stage2.block3",
        "stage3.block7",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientLite_medium(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="hard-swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:
    
    model = EfficientLite(
        feature_extractor=[ConvolutionBlock, Lite_EffiBlockS2],
        fusion_layer=Lite_EffiBlockS1,
        pyramid_pooling=None,
        filters=[24, 32, 64, 144, 288],
        num_blocks=[1, 1, 3, 7, 3],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientLite_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv6 lite version medium
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_m.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_m_finetune.py
    """

    model = EfficientLite_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1",
        "stage2.block3",
        "stage3.block7",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientLite_large(
    inputs=[640, 640, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="hard-swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False
) -> Model:
    
    model = EfficientLite(
        feature_extractor=[ConvolutionBlock, Lite_EffiBlockS2],
        fusion_layer=Lite_EffiBlockS1,
        pyramid_pooling=None,
        filters=[24, 48, 96, 192, 384],
        num_blocks=[1, 1, 3, 7, 3],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientLite_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv6 lite version large
        - In YOLOv6, feature extractor downsample percentage is: 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_l.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6_lite/yolov6_lite_l_finetune.py
    """

    model = EfficientLite_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block1",
        "stage2.block3",
        "stage3.block7",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientRep_nano(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=RepBlock,
        pyramid_pooling=CSPSPPF,
        filters=16,
        num_blocks=[1, 2, 4, 6, 2],
        csp_scale=1.0,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6n
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n_finetune.py
    """
    
    model = EfficientRep_nano(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientRep6_nano(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=RepBlock,
        pyramid_pooling=CSPSPPF,
        filters=[16, 32, 64, 128, 192, 256],
        num_blocks=[1, 2, 4, 6, 2, 2],
        csp_scale=1.0,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep6_nano_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6n6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6n6_finetune.py
    """
    
    model = EfficientRep6_nano(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientRep_small(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=RepBlock,
        pyramid_pooling=CSPSPPF,
        filters=32,
        num_blocks=[1, 2, 4, 6, 2],
        csp_scale=1.0,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6s
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s_finetune.py
    """
    
    model = EfficientRep_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientRep6_small(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=RepBlock,
        pyramid_pooling=CSPSPPF,
        filters=[32, 64, 128, 256, 384, 512],
        num_blocks=[1, 2, 4, 6, 2, 2],
        csp_scale=1.0,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep6_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6s6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6s6_finetune.py
    """
    
    model = EfficientRep6_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientRep_medium(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=BepC3,
        pyramid_pooling=SPPF,
        filters=48,
        num_blocks=[1, 4, 7, 11, 4],
        csp_scale=2/3,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6m
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m_finetune.py
    """
    
    model = EfficientRep_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientRep6_medium(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=RepVGGBlock,
        fusion_layer=BepC3,
        pyramid_pooling=SPPF,
        filters=[48, 96, 192, 384, 576, 768],
        num_blocks=[1, 4, 7, 11, 4, 4],
        csp_scale=2/3,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep6_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6m6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6m6_finetune.py
    """
    
    model = EfficientRep6_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientRep_large(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=BepC3,
        pyramid_pooling=SPPF,
        filters=64,
        num_blocks=[1, 6, 12, 18, 6],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    """
        - Used in YOLOv6l
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l_finetune.py
    """

    model = EfficientRep_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientRep6_large(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=BepC3,
        pyramid_pooling=SPPF,
        filters=[64, 128, 256, 512, 768, 1024],
        num_blocks=[1, 6, 12, 18, 6, 6],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientRep6_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6l6
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32, 64
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l6.py
            https://github.com/meituan/YOLOv6/blob/main/configs/yolov6l6_finetune.py
    """
    
    model = EfficientRep6_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )

    custom_layers = custom_layers or [
        "stem.block1",
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientMBLA_small(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=MBLABlock,
        pyramid_pooling=SPPF,
        filters=32,
        num_blocks=[1, 2, 4, 4, 2],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientMBLA_small_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6 MBLA version small
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6s_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6s_mbla_finetune.py
    """
    
    model = EfficientMBLA_small(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientMBLA_medium(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=MBLABlock,
        pyramid_pooling=SPPF,
        filters=48,
        num_blocks=[1, 2, 4, 4, 2],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientMBLA_medium_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6 MBLA version medium
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6m_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6m_mbla_finetune.py
    """
    
    model = EfficientMBLA_medium(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientMBLA_large(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=MBLABlock,
        pyramid_pooling=SPPF,
        filters=64,
        num_blocks=[1, 2, 4, 4, 2],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientMBLA_large_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6 MBLA version large
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6l_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6l_mbla_finetune.py
    """
    
    model = EfficientMBLA_large(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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


def EfficientMBLA_xlarge(
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
    deploy=False
) -> Model:
    
    model = EfficientRep(
        feature_extractor=ConvolutionBlock,
        fusion_layer=MBLABlock,
        pyramid_pooling=SPPF,
        filters=64,
        num_blocks=[1, 4, 8, 8, 4],
        csp_scale=0.5,
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
        drop_rate=drop_rate,
        deploy=deploy
    )
    return model


def EfficientMBLA_xlarge_backbone(
    inputs=[640, 640, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:
    
    """
        - Used in YOLOv6 MBLA version xlarge
        - In YOLOv6, feature extractor downsample percentage is: 4, 8, 16, 32
        - Reference:
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6x_mbla.py
            https://github.com/meituan/YOLOv6/blob/main/configs/mbla/yolov6x_mbla_finetune.py
    """
    
    model = EfficientMBLA_xlarge(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
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
