"""
  # Description:
    - The following table comparing the params of the RepVGG in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------------------------------
      |      Model Name       |    Train params   |    Test params    |
      |---------------------------------------------------------------|
      |       RepVGG-A0       |      9,132,616    |      8,309,384    |
      |---------------------------------------------------------------|
      |       RepVGG-A1       |     14,122,088    |     12,789,864    |
      |---------------------------------------------------------------|
      |       RepVGG-A2       |     28,253,160    |     25,499,944    |
      |---------------------------------------------------------------|
      |       RepVGG-B0       |     15,853,160    |     14,339,048    |
      |---------------------------------------------------------------|
      |       RepVGG-B1       |     57,483,112    |     51,829,480    |
      |---------------------------------------------------------------|
      |      RepVGG-B1g2      |     45,850,472    |     41,360,104    |
      |---------------------------------------------------------------|
      |      RepVGG-B1g4      |     40,034,152    |     36,125,416    |
      |---------------------------------------------------------------|
      |       RepVGG-B2       |     89,107,432    |     80,315,112    |
      |---------------------------------------------------------------|
      |      RepVGG-B2g2      |     70,931,432    |     63,956,712    |
      |---------------------------------------------------------------|
      |      RepVGG-B2g4      |     61,843,432    |     55,777,512    |
      |---------------------------------------------------------------|
      |       RepVGG-B3       |   123,185,256     |    110,960,872    |
      |---------------------------------------------------------------|
      |      RepVGG-B3g2      |    97,011,816     |     87,404,776    |
      |---------------------------------------------------------------|
      |      RepVGG-B3g4      |    83,925,096     |     75,626,728    |
       ---------------------------------------------------------------

  # Reference:
    - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)
    - Source: https://github.com/hoangthang1607/RepVGG-Tensorflow-2/tree/main

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


class RepVGGBlock(tf.keras.layers.Layer):
    ''' RepVGGBlock is a basic rep-style block, including training and deploy status
        This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=1,
        dilation=1,
        groups=1,
        activation='relu', 
        normalizer='batch-norm',
        training=False,
        *args, 
        **kwargs
    ):
        super(RepVGGBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.normalizer = normalizer
        self.training = training

        assert kernel_size == 3
        assert padding == 1

        self.padding_11 = padding - kernel_size // 2

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.nonlinearity = get_activation_from_name(self.activation, name=self.name + '_nonlinearity')

        if not self.training:
            self.rbr_reparam = Sequential([
                    ZeroPadding2D(padding=self.padding),
                    Conv2D(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding="valid",
                           dilation_rate=self.dilation,
                           groups=self.groups,
                           use_bias=True,
                           name=self.name + '_rbr_reparam')
            ])
        else:
            self.rbr_identity = get_normalizer_from_name(self.normalizer, name=self.name + '_rbr_identity') if self.filters == self.in_channels and self.strides == 1 else None
            
            self.rbr_dense = self.convolution_block(filters=self.filters,
                                                    kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    padding=self.padding,
                                                    groups=self.groups,
                                                    name=self.name + '_rbr_dense'
            )
            self.rbr_1x1 = self.convolution_block(filters=self.filters,
                                                  kernel_size=1,
                                                  strides=self.strides,
                                                  padding=self.padding_11,
                                                  groups=self.groups,
                                                  name=self.name + '_rbr_1x1'
            )

    def convolution_block(self, filters, kernel_size, strides, padding, groups=1, name=None):
        return Sequential([
                ZeroPadding2D(padding=padding, name="reppadding"),
                Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding="valid",
                       groups=groups,
                       use_bias=False,
                       name="repconv"),
                get_normalizer_from_name(self.normalizer, name="repbn")
        ], name=name
        )
        
    def call(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
            )

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Sequential):
            kernel = branch.get_layer("repconv").weights[0]
            running_mean = branch.get_layer("repbn").moving_mean
            running_var = branch.get_layer("repbn").moving_variance
            gamma = branch.get_layer("repbn").gamma
            beta = branch.get_layer("repbn").beta
            eps = branch.get_layer("repbn").epsilon
        else:
            assert isinstance(branch, BatchNormalization)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, input_dim, self.in_channels), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[1, 1, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
                )
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias


def RepVGG(num_blocks,
           width_multiplier=None,
           override_groups_map=None,
           training=False,
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           final_activation="softmax",
           classes=1000):

    def __get_list_strides(stride, num_block):
        return [stride] + [1] * (num_block - 1)
        
    assert len(width_multiplier) == 4
    override_groups_map = override_groups_map or dict()
    assert 0 not in override_groups_map 
               
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
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

    cur_layer_idx = 0
               
    x = RepVGGBlock(filters=min(64, int(64 * width_multiplier[0])),
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    activation='relu', 
                    normalizer='batch-norm',
                    training=training,
                    name=f"RepVGG_block_{cur_layer_idx + 1}")(img_input)
    cur_layer_idx += 1
               
    for st in __get_list_strides(2, num_blocks[0]):
        cur_groups = override_groups_map.get(cur_layer_idx, 1)
        x = RepVGGBlock(filters=int(64 * width_multiplier[0]),
                        kernel_size=3,
                        strides=st,
                        padding=1,
                        groups=cur_groups,
                        activation='relu', 
                        normalizer='batch-norm',
                        training=training,
                        name=f"RepVGG_block_{cur_layer_idx + 1}")(x)
        cur_layer_idx += 1

    for st in __get_list_strides(2, num_blocks[1]):
        cur_groups = override_groups_map.get(cur_layer_idx, 1)
        x = RepVGGBlock(filters=int(128 * width_multiplier[1]),
                        kernel_size=3,
                        strides=st,
                        padding=1,
                        groups=cur_groups,
                        activation='relu', 
                        normalizer='batch-norm',
                        training=training,
                        name=f"RepVGG_block_{cur_layer_idx + 1}")(x)
        cur_layer_idx += 1

    for st in __get_list_strides(2, num_blocks[2]):
        cur_groups = override_groups_map.get(cur_layer_idx, 1)
        x = RepVGGBlock(filters=int(256 * width_multiplier[2]),
                        kernel_size=3,
                        strides=st,
                        padding=1,
                        groups=cur_groups,
                        activation='relu', 
                        normalizer='batch-norm',
                        training=training,
                        name=f"RepVGG_block_{cur_layer_idx + 1}")(x)
        cur_layer_idx += 1

    for st in __get_list_strides(2, num_blocks[3]):
        cur_groups = override_groups_map.get(cur_layer_idx, 1)
        x = RepVGGBlock(filters=int(512 * width_multiplier[3]),
                        kernel_size=3,
                        strides=st,
                        padding=1,
                        groups=cur_groups,
                        activation='relu', 
                        normalizer='batch-norm',
                        training=training,
                        name=f"RepVGG_block_{cur_layer_idx + 1}")(x)

    if include_top:
        # Classification block
        x = tfa.layers.AdaptiveAveragePooling2D(output_size=1)(x)
        x = Flatten(name='flatten')(x)
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
    if num_blocks == [2, 4, 14, 1]:
        if width_multiplier == [0.75, 0.75, 0.75, 2.5]:
            i = 0
        elif width_multiplier == [1, 1, 1, 2.5]:
            i = 1
        elif width_multiplier == [1.5, 1.5, 1.5, 2.75]:
            i = 2
        else:
            i = ''
        model = Model(inputs, x, name=f'RepVGG-A{i}')
        
    elif num_blocks == [4, 6, 16, 1]:
        if width_multiplier == [1, 1, 1, 2.5]:
            i = 0
        elif width_multiplier == [2, 2, 2, 4]:
            i = 1
        elif width_multiplier == [2.5, 2.5, 2.5, 5]:
            i = 2
        elif width_multiplier == [3, 3, 3, 5]:
            i = 3
        else:
            i = ''

        if override_groups_map == g2_map:
            g = 2
        elif override_groups_map == g4_map:
            g = 4
        else:
            g = ''

        if g != '':
            model = Model(inputs, x, name=f'RepVGG-B{i}g{g}')
        else:
            model = Model(inputs, x, name=f'RepVGG-B{i}')
    else:
        model = Model(inputs, x, name='RepVGG')

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


def RepVGG_A0(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[0.75, 0.75, 0.75, 2.5],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_A1(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[1, 1, 1, 2.5],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_A2(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[2, 4, 14, 1],
                   width_multiplier=[1.5, 1.5, 1.5, 2.75],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B0(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[1, 1, 1, 2.5],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B1(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2, 2, 2, 4],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B1g2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000,
                training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2, 2, 2, 4],
                   override_groups_map=g2_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B1g4(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000,
                training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2, 2, 2, 4],
                   override_groups_map=g4_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B2(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2.5, 2.5, 2.5, 5],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B2g2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000,
                training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2.5, 2.5, 2.5, 5],
                   override_groups_map=g2_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B2g4(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000,
                training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[2.5, 2.5, 2.5, 5],
                   override_groups_map=g4_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B3(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[3, 3, 3, 5],
                   override_groups_map=None,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B3g2(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[3, 3, 3, 5],
                   override_groups_map=g2_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def RepVGG_B3g4(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              training=False) -> Model:
    
    model = RepVGG(num_blocks=[4, 6, 16, 1],
                   width_multiplier=[3, 3, 3, 5],
                   override_groups_map=g4_map,
                   training=training,
                   include_top=include_top,
                   weights=weights, 
                   input_tensor=input_tensor, 
                   input_shape=input_shape, 
                   pooling=pooling, 
                   final_activation=final_activation,
                   classes=classes)
    return model


def repvgg_reparameter(model: tf.keras.Model, structure, input_shape=(224, 224, 3), classes=1000, save_path=None):
    deploy_model = structure(input_shape=input_shape, classes=classes, training=False)
    for layer, deploy_layer in zip(model.layers, deploy_model.layers):
        if hasattr(layer, "repvgg_convert"):
            kernel, bias = layer.repvgg_convert()
            deploy_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.Sequential):
            assert isinstance(deploy_layer, tf.keras.Sequential)
            for sub_layer, deploy_sub_layer in zip(layer.layers, deploy_layer.layers):
                kernel, bias = sub_layer.repvgg_convert()
                deploy_sub_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    if save_path is not None:
        deploy_model.save_weights(save_path)

    return deploy_model
