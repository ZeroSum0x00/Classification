"""
  # Description:
    - The following table comparing the params of the Inception Resnet v1 in Tensorflow on 
    size 299 x 299 x 3:

       ---------------------------------------------
      |         Model Name        |     Params      |
      |---------------------------------------------|
      |    Inception Resnet v1    |   136,038,104   |
       ---------------------------------------------

  # Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
    - Source: https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v2.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def convolution_block(inputs, 
                      filters, 
                      kernel_size, 
                      strides=(1, 1), 
                      padding='same', 
                      activation='relu',
                      name=None):
                          
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               strides=strides, 
               padding=padding, 
               name=name + '_conv')(inputs)

    if activation is not None:
        x = get_activation_from_name(activation, name=name + '_activ')(x)
    return x


def stem_block(inputs, activation="relu", normalizer='batch-norm'):
    x = convolution_block(inputs, 32, (3, 3), (2, 2), padding='valid', name='stem_b1')
    x = convolution_block(x, 32, (3, 3), padding='valid', name='stem_b2')
    x = convolution_block(x, 64, (3, 3), name='stem_b3')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stem_maxpool')(x)
    x = convolution_block(x, 80, (1, 1), name='stem_b4')
    x = convolution_block(x, 192, (3, 3), name='stem_b5')
    x = convolution_block(x, 256, (3, 3), (2, 2), name='stem_b6')
    x = get_normalizer_from_name(normalizer, name='stem_bn')(x)
    x = get_activation_from_name(activation, name='stem_activ')(x)
    return x


def inception_resnet_A(inputs, scale_residual=True, activation="relu", normalizer='batch-norm', prefix=None):
    shortcut = inputs
    
    branch1 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b12")

    branch2 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b21")
    branch2 = convolution_block(branch2, 32, (3, 3), name=f"inceptionA_step{prefix}_b22")

    branch3 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b31")
    branch3 = convolution_block(branch3, 32, (3, 3), name=f"inceptionA_step{prefix}_b32")
    branch3 = convolution_block(branch3, 32, (3, 3), name=f"inceptionA_step{prefix}_b33")

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f'inceptionA_step{prefix}_submerged')
    x = convolution_block(x, 256, (1, 1), activation=None, name=f'inceptionA_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionA_step{prefix}_scale')(x)
        
    out = add([shortcut, x], name=f'inceptionA_step{prefix}_merged')
    out = get_normalizer_from_name(normalizer, name=f'inceptionA_step{prefix}_bn')(out)
    out = get_activation_from_name(activation, name=f'inceptionA_step{prefix}_activ')(out)
    return out


def reduction_A(inputs, k=192, l=192, m=256, n=384, activation="relu", normalizer='batch-norm'):
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b11")(inputs)

    branch2 = convolution_block(inputs, n, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b21")

    branch3 = convolution_block(inputs, k, (1, 1), name=f"incep_reductionA_b31")
    branch3 = convolution_block(branch3, l, (3, 3), name=f"incep_reductionA_b32")
    branch3 = convolution_block(branch3, m, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b33")

    out = concatenate([branch1, branch2, branch3], axis=-1, name=f'incep_reductionA_merged')
    out = get_normalizer_from_name(normalizer, name=f'incep_reductionA_bn')(out)
    out = get_activation_from_name(activation, name=f'incep_reductionA_activ')(out)
    return out


def inception_resnet_B(inputs, scale_residual=True, activation="relu", normalizer='batch-norm', prefix=None):
    shortcut = inputs

    branch1 = convolution_block(inputs, 128, (1, 1), name=f"inceptionB_step{prefix}_b12")

    branch2 = convolution_block(inputs, 128, (1, 1), name=f"inceptionB_step{prefix}_b21")
    branch2 = convolution_block(branch2, 128, (1, 7), name=f"inceptionB_step{prefix}_b22")
    branch2 = convolution_block(branch2, 128, (7, 1), name=f"inceptionB_step{prefix}_b23")

    x = concatenate([branch1, branch2], axis=-1, name=f'inceptionB_step{prefix}_submerged')
    x = convolution_block(x, 896, (1, 1), activation=None, name=f'inceptionB_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionB_step{prefix}_scale')(x)

    out = add([shortcut, x], name=f'inceptionB_step{prefix}_merged')
    out = get_normalizer_from_name(normalizer, name=f'inceptionB_step{prefix}_bn')(out)
    out = get_activation_from_name(activation, name=f'inceptionB_step{prefix}_activ')(out)
    return out


def reduction_B(inputs, activation="relu", normalizer='batch-norm'):
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b11")(inputs)

    branch2 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b21")
    branch2 = convolution_block(inputs, 384, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b22")

    branch3 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b31")
    branch3 = convolution_block(branch3, 256, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b32")

    branch4 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b41")
    branch4 = convolution_block(branch4, 256, (1, 1), name=f"incep_reductionB_b42")
    branch4 = convolution_block(branch4, 256, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b33")
    
    out = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f'incep_reductionB_merged')
    out = get_normalizer_from_name(normalizer, name=f'incep_reductionB_bn')(out)
    out = get_activation_from_name(activation, name=f'incep_reductionB_activ')(out)
    return out


def inception_resnet_C(inputs, scale_residual=True, activation="relu", normalizer='batch-norm', prefix=None):
    shortcut = inputs
    
    branch1 = convolution_block(inputs, 128, (1, 1), name=f"inceptionC_step{prefix}_b12")

    branch2 = convolution_block(inputs, 192, (1, 1), name=f"inceptionC_step{prefix}_b21")
    branch2 = convolution_block(branch2, 192, (1, 3), name=f"inceptionC_step{prefix}_b22")
    branch2 = convolution_block(branch2, 192, (3, 1), name=f"inceptionC_step{prefix}_b23")

    x = concatenate([branch1, branch2], axis=-1, name=f'inceptionC_step{prefix}_submerged')
    x = convolution_block(x, 1792, (1, 1), activation=None, name=f'inceptionC_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionC_step{prefix}_scale')(x)
        
    out = add([shortcut, x], name=f'inceptionC_step{prefix}_merged')
    out = get_normalizer_from_name(normalizer, name=f'inceptionC_step{prefix}_bn')(out)
    out = get_activation_from_name(activation, name=f'inceptionC_step{prefix}_activ')(out)
    return out


def Inception_Resnet_v1(depths=[5, 10, 5],
                        scale_residual=True,
                        include_top=True, 
                        weights='imagenet',
                        input_tensor=None, 
                        input_shape=None,
                        pooling=None,
                        final_activation='softmax',
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

    # Stem block
    x = stem_block(img_input)

    # Inception-A
    for i in range(depths[0]):
        x = inception_resnet_A(x, scale_residual=scale_residual, prefix=str(i+1))

    # Reduction-A
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # Inception-B
    for i in range(depths[1]):
        x = inception_resnet_B(x, scale_residual=scale_residual, prefix=str(i+1))

    # Reduction-B
    x = reduction_B(x)
                  
    # Inception-C
    for i in range(depths[2]):
        x = inception_resnet_C(x, scale_residual=scale_residual, prefix=str(i+1))

    if include_top:
        # Classification block
        x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='same', name='avg_pooling')(x)
        x = Dropout(rate=0.8, name='dropout')(x)
        x = Flatten(name='flatten')(x)
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
    model = Model(inputs, x, name='Inception-Resnet-v1')

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