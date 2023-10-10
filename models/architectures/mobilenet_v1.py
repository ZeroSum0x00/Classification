"""
  # Description:
    - The following table comparing the params of the MobileNet v1 in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------------
      |         Model Name       |    Params       |
      |--------------------------------------------|
      |     0.25 MobileNetV1     |      475,544    |
      |--------------------------------------------|
      |     0.5 MobileNetV1      |    1,342,536    |
      |--------------------------------------------|
      |     0.75 MobileNetV1     |    2,601,976    |
      |--------------------------------------------|
      |     1.0 MobileNetV1      |    4,253,864    |
       --------------------------------------------

  # Reference:
    - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import ReLU6
from utils.model_processing import _obtain_input_shape


def stem_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), alpha=1):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='stem_pad')(inputs)
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size,
               strides=strides,
               padding='valid',
               use_bias=False,
               name='stem_conv')(x)
    x = BatchNormalization(axis=-1, name='stem_bn')(x)
    x = ReLU6(name='stem_relu')(x)
    return x


def depthwise_separable_convolutional(inputs, out_dim, strides=(1, 1), alpha=1, depth_multiplier=1, prefix=None):
    pointwise_filters = int(out_dim * alpha)

    if strides != (1, 1):
        inputs = ZeroPadding2D(((0, 1), (0, 1)), name=f'conv_pad_{prefix}')(inputs)
        padding = 'valid'
    else:
        padding = 'same'

    # Depthwise layer
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        padding=padding,
                        depth_multiplier=depth_multiplier,
                        use_bias=False,
                        name=f'conv_dw_{prefix}')(inputs)
    x = BatchNormalization(axis=-1, name=f'conv_dw_{prefix}_bn')(x)
    x = ReLU6(name=f'conv_dw_{prefix}_activ')(x)

    # Pointwise
    x = Conv2D(filters=pointwise_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               name=f'conv_pw_{prefix}')(x)
    x = BatchNormalization(axis=-1, name=f'conv_pw_{prefix}_bn')(x)
    x = ReLU6(name=f'conv_pw_{prefix}_activ')(x)
    return x
    

def MobileNet_v1(alpha=1,
                 depth_multiplier=1,
                 include_top=True, 
                 weights='imagenet',
                 input_tensor=None, 
                 input_shape=None,
                 pooling=None,
                 final_activation='softmax',
                 classes=1000,
                 drop_rate=1e-3):
                  
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=75,
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

    bs, h, w, c = img_input.shape
    if weights == 'imagenet':
        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')
            
        if h != w and h not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '128, 160, 192 or 224 only.')
            
    x = stem_block(img_input, 32, (3, 3), (2, 2), alpha)

    x = depthwise_separable_convolutional(x, 64, (1, 1), alpha, depth_multiplier, prefix=1)

    x = depthwise_separable_convolutional(x, 128, (2, 2), alpha, depth_multiplier, prefix=2)
    x = depthwise_separable_convolutional(x, 128, (1, 1), alpha, depth_multiplier, prefix=3)

    x = depthwise_separable_convolutional(x, 256, (2, 2), alpha, depth_multiplier, prefix=4)
    x = depthwise_separable_convolutional(x, 256, (1, 1), alpha, depth_multiplier, prefix=5)

    x = depthwise_separable_convolutional(x, 512, (2, 2), alpha, depth_multiplier, prefix=6)
    for i in range(5):
        x = depthwise_separable_convolutional(x, 512, (1, 1), alpha, depth_multiplier, prefix=i+7)

    x = depthwise_separable_convolutional(x, 1024, (2, 2), alpha, depth_multiplier, prefix=12)
    x = depthwise_separable_convolutional(x, 1024, (1, 1), alpha, depth_multiplier, prefix=13)

    if include_top:
        # Classification block
        out_dim = 1 if classes == 2 else classes
        x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = Reshape((1, 1, -1), name='reshape_layer1')(x)
        x = Dropout(drop_rate, name='dropout')(x)
        x = Conv2D(filters=out_dim, 
                   kernel_size=(1, 1), 
                   padding='same', 
                   name='conv_preds')(x)
        x = Reshape((out_dim,), name='reshape_layer2')(x)
        x = Activation(final_activation, name='final_activation')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_max_pooling')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name=f'{float(alpha)} MobileNetV1-{h}')

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