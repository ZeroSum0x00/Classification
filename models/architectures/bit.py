"""
  # Description:
    - The following table comparing the params of the Big Transfer (BiT) in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     BiT_S_R50x1     |   25,549,352    |
      |---------------------|-----------------|
      |     BiT_S_R50x3     |   217,177,128   |
      |---------------------|-----------------|
      |     BiT_M_R50x1     |   25,549,352    |
      |---------------------|-----------------|
      |     BiT_M_R50x3     |   217,177,128   |
      |---------------------|-----------------|
      |     BiT_S_R101x1    |   44,541,480    |
      |---------------------|-----------------|
      |     BiT_S_R101x3    |   387,792,936   |
      |---------------------|-----------------|
      |     BiT_M_R101x1    |   44,541,480    |
      |---------------------|-----------------|
      |     BiT_M_R101x3    |   387,792,936   |
      |---------------------|-----------------|
      |     BiT_S_R152x4    |   936,258,856   |
      |---------------------|-----------------|
      |     BiT_M_R152x4    |   936,258,856   |
       ---------------------------------------

  # Reference:
    - [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/pdf/1912.11370.pdf)
    - Source: https://github.com/google-research/big_transfer
    - Note: The S and M variants of the model are the same but trained on the ImageNet-21k and JFT-300M datasets respectively
    
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import GroupNormalization
from utils.model_processing import _obtain_input_shape


class PaddingFromKernelSize(tf.keras.layers.Layer):
  """Layer that adds padding to an image taking into a given kernel size."""

  def __init__(self, kernel_size, **kwargs):
    super(PaddingFromKernelSize, self).__init__(**kwargs)
    if isinstance(kernel_size, int):
        pad_total = kernel_size - 1
    else:
        pad_total = kernel_size[0] - 1
    self._pad_beg = pad_total // 2
    self._pad_end = pad_total - self._pad_beg

  def call(self, x):
    padding = [
        [0, 0],
        [self._pad_beg, self._pad_end],
        [self._pad_beg, self._pad_end],
        [0, 0]]
    return tf.pad(x, padding)


def stem_block(inputs, filters, conv_size=(7, 7), conv_stride=(2, 2), pool_size=(3, 3), pool_stride=(2, 2)):
    x = PaddingFromKernelSize(kernel_size=conv_size)(inputs)
    x = Conv2D(filters=filters,
               kernel_size=conv_size,
               strides=conv_stride,
               use_bias=False)(x)
    x = PaddingFromKernelSize(kernel_size=pool_size)(x)
    x = MaxPool2D(pool_size=pool_size, strides=pool_stride, padding="valid")(x)
    return x

    
def bottleneck_unit_v2(inputs, num_filters, stride=2):
    shortcut = inputs
    x = GroupNormalization()(inputs)
    x = ReLU()(x)

    if (stride > 1) or (4 * num_filters != inputs.shape[-1]):
        shortcut = Conv2D(filters=4 * num_filters,
                          kernel_size=1,
                          strides=stride,
                          use_bias=False,
                          padding="valid")(x)
    x = Conv2D(filters=num_filters,
               kernel_size=1,
               use_bias=False,
               padding="valid")(x)
    x = GroupNormalization()(x)
    x = ReLU()(x)
    x = PaddingFromKernelSize(kernel_size=3)(x)
    x = Conv2D(filters=num_filters,
               kernel_size=3,
               strides=stride,
               use_bias=False,
               padding="valid")(x)
    x = GroupNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=4 * num_filters,
               kernel_size=1,
               use_bias=False,
               padding="valid")(x)
    return add([x, shortcut])


def resnet_block(inputs, filters, stride, iter):
    x = inputs
    for i in range(1, iter + 1):
        x = bottleneck_unit_v2(x, filters, (stride if i == 1 else 1))
    return x

     
def ResnetV2(layers,
             strides,
             filter_downsample_factor,
             model_variant,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
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

    filters = tuple(16 * filter_downsample_factor * 2**b for b in range(len(layers)))
    
    x = stem_block(img_input, 64)
                 
    for filter, stride, layer in zip(filters, strides, layers):
        x = resnet_block(x, filter, stride, layer)

    x = GroupNormalization()(x)
    x = ReLU()(x)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D()(x)
        x = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(x)
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
    if layers == (3, 4, 6, 3):
        model = Model(inputs, x, name=f'BiT-{model_variant.upper()}-R50x{filter_downsample_factor//4}')
    elif layers == (3, 4, 23, 3):
        model = Model(inputs, x, name=f'BiT-{model_variant.upper()}-R101x{filter_downsample_factor//4}')
    elif layers == (3, 8, 36, 3):
        model = Model(inputs, x, name=f'BiT-{model_variant.upper()}-R152x{filter_downsample_factor//4}')
    else:
        model = Model(inputs, x, name=f'BiT-{model_variant.upper()}')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if layers == (3, 4, 6, 3):
                if filter_downsample_factor / 4 == 1:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
                elif filter_downsample_factor / 4 == 3:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
            elif layers == (3, 4, 23, 3):
                if filter_downsample_factor / 4 == 1:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
                elif filter_downsample_factor / 4 == 3:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
            elif layers == (3, 8, 36, 3):
                if filter_downsample_factor / 4 == 4:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
        else:
            if layers == (3, 4, 6, 3):
                if filter_downsample_factor / 4 == 1:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
                elif filter_downsample_factor / 4 == 3:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
            elif layers == (3, 4, 23, 3):
                if filter_downsample_factor / 4 == 1:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
                elif filter_downsample_factor / 4 == 3:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
            elif layers == (3, 8, 36, 3):
                if filter_downsample_factor / 4 == 4:
                    if model_variant.upper() == "S":
                        weights_path = None
                    elif model_variant.upper() == "M":
                        weights_path = None
                        
        try:
            if weights_path is not None:
                model.load_weights(weights_path)
        except:
            raise Exception("Imagenet weights is not have this variant!")
            
    elif weights is not None:
        model.load_weights(weights)
        
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



def BiT_S_R50x1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 6, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=4,
                     model_variant='S',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_S_R50x3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 6, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=12,
                     model_variant='S',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model

                    
def BiT_M_R50x1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 6, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=4,
                     model_variant='M',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_M_R50x3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 6, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=12,
                     model_variant='M',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_S_R101x1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 23, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=4,
                     model_variant='S',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_S_R101x3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 23, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=12,
                     model_variant='S',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model

                    
def BiT_M_R101x1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 23, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=4,
                     model_variant='M',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_M_R101x3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 4, 23, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=12,
                     model_variant='M',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_S_R152x4(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 8, 36, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=16,
                     model_variant='S',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def BiT_M_R152x4(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = ResnetV2(layers=(3, 8, 36, 3),
                     strides=(1, 2, 2, 2),
                     filter_downsample_factor=16,
                     model_variant='M',
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model
