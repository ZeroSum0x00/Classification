import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, BatchNormalization
)
from tensorflow.keras.regularizers import l2

from . import get_activation_from_name, get_normalizer_from_name
from models.architectures.vgg.repvgg import RepVGGBlock



class QARepVGGBlockV1(RepVGGBlock):
    
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="relu",
        normalizer="batch-norm",
        training=False,
        *args, **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            normalizer=normalizer,
            training=training,
            *args, **kwargs
        )
    
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.nonlinearity = get_activation_from_name(self.activation, name=self.name + "_nonlinearity")

        if not self.training:
            self.rbr_reparam = Sequential([
                    ZeroPadding2D(padding=self.padding),
                    Conv2D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding="valid",
                        dilation_rate=self.dilation,
                        groups=self.groups,
                        use_bias=True,
                        name= f"{self.name}_rbr_reparam"
                    )
            ])
        else:
            self.bn = get_normalizer_from_name(self.normalizer)
            self.rbr_identity = LinearLayer() if (self.filters == self.in_channels and stride == 1) else None
            
            self.rbr_1x1 = self.convolution_block(
                filters=self.filters,
                kernel_size=1,
                strides=self.strides,
                padding=self.padding_11,
                groups=self.groups,
                name=f"{self.name}_rbr_1x1"
            )
            self.rbr_dense = self.convolution_block(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                groups=self.groups,
                name=f"{self.name}_rbr_dense"
            )

    def call(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )


    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weights[0])
        bias = bias3x3

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, 3, 3, input_dim), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, 1, 1, i % input_dim] = 1
            kernel = kernel + kernel_value
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, BatchNormalization)
        running_mean = branch.moving_mean - bias # remove bias
        running_var = branch.moving_variance
        gamma = branch.gamma
        beta = branch.beta
        eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std


class QARepVGGBlockV2(RepVGGBlock):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="relu",
        normalizer="batch-norm",
        training=False,
        *args, **kwargs
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            normalizer=normalizer,
            training=training,
            *args, **kwargs
        )
    
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.nonlinearity = get_activation_from_name(self.activation, name=self.name + "_nonlinearity")

        if not self.training:
            self.rbr_reparam = Sequential([
                    ZeroPadding2D(padding=self.padding),
                    Conv2D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding="valid",
                        dilation_rate=self.dilation,
                        groups=self.groups,
                        use_bias=True,
                        name=f"{self.name}_rbr_reparam"
                    )
            ])
        else:
            self.bn = get_normalizer_from_name(self.normalizer)
            self.rbr_identity = LinearLayer() if (self.filters == self.in_channels and stride == 1) else None
            
            self.rbr_1x1 = self.convolution_block(
                filters=self.filters,
                kernel_size=1,
                strides=self.strides,
                padding=self.padding_11,
                groups=self.groups,
                name=f"{self.name}_rbr_1x1"
            )
            
            self.rbr_dense = self.convolution_block(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                groups=self.groups,
                name=f"{self.name}_rbr_dense"
            )
            
            self.rbr_avg = AveragePooling2D(
                pool_size=self.kernel_size,
                strides=self.strides,
                padding="same",
            ) if (self.filters == self.in_channels and stride == 1) else None
            
    def call(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        if self.rbr_avg is None:
            avg_out = 0
        else:
            avg_out = self.rbr_avg(inputs)

        return self.nonlinearity(
            self.bn(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out + avg_out)
        )

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weights[0])
        bias = bias3x3
        if self.rbr_avg is not None:
            kernel_avg = self._avg_to_3x3_tensor(self.rbr_avg)
            kernel = kernel + kernel_avg
            
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, 3, 3, input_dim), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, 1, 1, i % input_dim] = 1
            kernel = kernel + kernel_value
        return kernel, bias

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.pool_size
        input_dim = channels // groups
        k = tf.zeros((channels, kernel_size, kernel_size, input_dim))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k
        
    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, BatchNormalization)
        running_mean = branch.moving_mean - bias # remove bias
        running_var = branch.moving_variance
        gamma = branch.gamma
        beta = branch.beta
        eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std
    