"""
  # Description:
    - The following table comparing the params of the MobileNet v2 in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------------
      |         Model Name       |    Params       |
      |--------------------------------------------|
      |     0.35 MobileNetV2     |    1,529,768    |
      |--------------------------------------------|
      |     0.5 MobileNetV2      |    1,987,224    |
      |--------------------------------------------|
      |     0.75 MobileNetV2     |    2,663,064    |
      |--------------------------------------------|
      |     1.0 MobileNetV2      |    3,538,984    |
      |--------------------------------------------|
      |     1.3 MobileNetV2      |    5,431,116    |
      |--------------------------------------------|
      |     1.4 MobileNetV2      |    6,156,440    |
       --------------------------------------------

  # Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape, correct_pad
from utils.auxiliary_processing import make_divisible


def stem_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), alpha=1, activation="relu6", normalizer='batch-norm'):
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3), name='stem_pad')(inputs)
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size,
               strides=strides,
               padding='valid',
               use_bias=False,
               name='stem_conv')(x)
    x = get_normalizer_from_name(normalizer, name='stem_bn')(x)
    x = get_activation_from_name(activation, name='stem_relu')(x)
    return x


def inverted_residual_block(inputs, out_dim, strides=(1, 1), expansion=1, alpha=1, activation="relu6", normalizer='batch-norm', block_id=0):
    bs, h, w, c = inputs.shape
    pointwise_filters = make_divisible(int(out_dim * alpha), 8)

    x = inputs

    if block_id:
        prefix = f'block{block_id}_'
        x = Conv2D(expansion * c,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same',
                   use_bias=False,
                   activation=None,
                   name=prefix + 'expand')(x)
        x = get_normalizer_from_name(normalizer, 
                                    epsilon=1e-3,
                                    momentum=0.999,
                                    name=prefix + 'expand_bn')(x)
        x = get_activation_from_name(activation, name=prefix + 'expand_activ')(x)
    else:
        prefix = 'expanded_conv_'

    if strides == (2, 2):
        x = ZeroPadding2D(padding=correct_pad(x, 3), name=prefix + 'pad')(x)

    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        activation=None,
                        use_bias=False,
                        padding='same' if strides == (1, 1) else 'valid',
                        name=prefix + 'depthwise')(x)
    x = get_normalizer_from_name(normalizer, 
                                epsilon=1e-3,
                                momentum=0.999,
                                name=prefix + 'depthwise_BN')(x)
    x = get_activation_from_name(activation, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = get_normalizer_from_name(normalizer, 
                                epsilon=1e-3,
                                momentum=0.999,
                                name=prefix + 'project_BN')(x)
    
    if c == pointwise_filters and strides == (1, 1):
        return add([inputs, x])
        
    return x
    

def MobileNet_v2(alpha=1,
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

    x = inverted_residual_block(x, out_dim=16, strides=(1, 1), expansion=1, alpha=alpha, block_id=0)

    x = inverted_residual_block(x, out_dim=24, strides=(2, 2), expansion=6, alpha=alpha, block_id=1)
    x = inverted_residual_block(x, out_dim=24, strides=(1, 1), expansion=6, alpha=alpha, block_id=2)

    x = inverted_residual_block(x, out_dim=32, strides=(2, 2), expansion=6, alpha=alpha, block_id=3)
    x = inverted_residual_block(x, out_dim=32, strides=(1, 1), expansion=6, alpha=alpha, block_id=4)
    x = inverted_residual_block(x, out_dim=32, strides=(1, 1), expansion=6, alpha=alpha, block_id=5)

    x = inverted_residual_block(x, out_dim=64, strides=(2, 2), expansion=6, alpha=alpha, block_id=6)
    x = inverted_residual_block(x, out_dim=64, strides=(1, 1), expansion=6, alpha=alpha, block_id=7)
    x = inverted_residual_block(x, out_dim=64, strides=(1, 1), expansion=6, alpha=alpha, block_id=8)
    x = inverted_residual_block(x, out_dim=64, strides=(1, 1), expansion=6, alpha=alpha, block_id=9)

    x = inverted_residual_block(x, out_dim=96, strides=(1, 1), expansion=6, alpha=alpha, block_id=10)
    x = inverted_residual_block(x, out_dim=96, strides=(1, 1), expansion=6, alpha=alpha, block_id=11)
    x = inverted_residual_block(x, out_dim=96, strides=(1, 1), expansion=6, alpha=alpha, block_id=12)

    x = inverted_residual_block(x, out_dim=160, strides=(2, 2), expansion=6, alpha=alpha, block_id=13)
    x = inverted_residual_block(x, out_dim=160, strides=(1, 1), expansion=6, alpha=alpha, block_id=14)
    x = inverted_residual_block(x, out_dim=160, strides=(1, 1), expansion=6, alpha=alpha, block_id=15)
                  
    x = inverted_residual_block(x, out_dim=320, strides=(1, 1), expansion=6, alpha=alpha, block_id=16)
                  
    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               name='conv1')(x)
    x = get_normalizer_from_name('batch-norm', 
                                epsilon=1e-3,
                                momentum=0.999,
                                name='Conv_1_bn')(x)
    x = get_activation_from_name('relu6', name='out_relu')(x)
                     
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
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
    model = Model(inputs, x, name=f'{float(alpha)} MobileNetV2-{h}')

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