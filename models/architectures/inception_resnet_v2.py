"""
  # Description:
    - The following table comparing the params of the Inception Resnet v2 in Tensorflow on 
    size 299 x 299 x 3:

       ---------------------------------------------
      |         Model Name        |     Params      |
      |---------------------------------------------|
      |    Inception Resnet v2    |   151,451,288   |
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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file
from .inception_resnet_v1 import convolution_block
from utils.model_processing import _obtain_input_shape


def stem_block(inputs):
    x = convolution_block(inputs, 32, (3, 3), (2, 2), padding='valid', name='stem_b11')
    x = convolution_block(x, 32, (3, 3), padding='valid', name='stem_b12')
    x = convolution_block(x, 64, (3, 3), name='stem_b13')
    
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stem_b21')(x)
    branch2 = convolution_block(x, 96, (3, 3), strides=(2, 2), padding='valid', name='stem_b22')
    
    x = concatenate([branch1, branch2], axis=-1, name='stem_merged1')
    
    branch1 = convolution_block(x, 64, (1, 1), name='stem_b31')
    branch1 = convolution_block(branch1, 96, (3, 3), padding='valid', name='stem_b32')

    branch2 = convolution_block(x, 64, (1, 1), name='stem_b41')
    branch2 = convolution_block(branch2, 64, (7, 1), name='stem_b42')
    branch2 = convolution_block(branch2, 64, (1, 7), name='stem_b43')
    branch2 = convolution_block(branch2, 96, (3, 3), padding='valid', name='stem_b44')

    x = concatenate([branch1, branch2], axis=-1, name='stem_merged2')

    branch1 = convolution_block(x, 192, (3, 3), strides=(2, 2), padding='valid', name='stem_b51')    
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='stem_b52')(x)
    
    x = concatenate([branch1, branch2], axis=-1, name='stem_merged3')
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x


def inception_resnet_A(inputs, scale_residual=True, prefix=None):
    shortcut = inputs
    
    branch1 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b12")

    branch2 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b21")
    branch2 = convolution_block(branch2, 32, (3, 3), name=f"inceptionA_step{prefix}_b22")

    branch3 = convolution_block(inputs, 32, (1, 1), name=f"inceptionA_step{prefix}_b31")
    branch3 = convolution_block(branch3, 48, (3, 3), name=f"inceptionA_step{prefix}_b32")
    branch3 = convolution_block(branch3, 64, (3, 3), name=f"inceptionA_step{prefix}_b33")

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f'inceptionA_step{prefix}_submerged')
    x = convolution_block(x, 384, (1, 1), activation=None, name=f'inceptionA_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionA_step{prefix}_scale')(x)
        
    out = add([shortcut, x], name=f'inceptionA_step{prefix}_merged')
    out = BatchNormalization(axis=-1, name=f'inceptionA_step{prefix}_bn')(out)
    out = Activation("relu", name=f'inceptionA_step{prefix}_activ')(out)
    return out


def reduction_A(inputs, k=192, l=224, m=256, n=384):
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b11")(inputs)

    branch2 = convolution_block(inputs, n, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b21")

    branch3 = convolution_block(inputs, k, (1, 1), name=f"incep_reductionA_b31")
    branch3 = convolution_block(branch3, l, (3, 3), name=f"incep_reductionA_b32")
    branch3 = convolution_block(branch3, m, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionA_b33")

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f'incep_reductionA_merged')
    out = BatchNormalization(axis=-1, name=f'incep_reductionA_bn')(x)
    out = Activation("relu", name=f'incep_reductionA_activ')(out)
    return x


def inception_resnet_B(inputs, scale_residual=True, prefix=None):
    shortcut = inputs
    
    branch1 = convolution_block(inputs, 192, (1, 1), name=f"inceptionB_step{prefix}_b12")

    branch2 = convolution_block(inputs, 128, (1, 1), name=f"inceptionB_step{prefix}_b21")
    branch2 = convolution_block(branch2, 160, (1, 7), name=f"inceptionB_step{prefix}_b22")
    branch2 = convolution_block(branch2, 192, (7, 1), name=f"inceptionB_step{prefix}_b23")

    x = concatenate([branch1, branch2], axis=-1, name=f'inceptionB_step{prefix}_submerged')
    x = convolution_block(x, 1152, (1, 1), activation=None, name=f'inceptionB_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionB_step{prefix}_scale')(x)
        
    out = add([shortcut, x], name=f'inceptionB_step{prefix}_merged')
    out = BatchNormalization(axis=-1, name=f'inceptionB_step{prefix}_bn')(out)
    out = Activation("relu", name=f'inceptionB_step{prefix}_activ')(out)
    return out


def reduction_B(inputs):
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b11")(inputs)

    branch2 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b21")
    branch2 = convolution_block(inputs, 384, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b22")

    branch3 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b31")
    branch3 = convolution_block(branch3, 288, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b32")

    branch4 = convolution_block(inputs, 256, (1, 1), name=f"incep_reductionB_b41")
    branch4 = convolution_block(branch4, 288, (1, 1), name=f"incep_reductionB_b42")
    branch4 = convolution_block(branch4, 320, (3, 3), strides=(2, 2), padding="valid", name=f"incep_reductionB_b33")
    
    x = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f'incep_reductionB_merged')
    out = BatchNormalization(axis=-1, name=f'incep_reductionB_bn')(x)
    out = Activation("relu", name=f'incep_reductionB_activ')(out)
    return x


def inception_resnet_C(inputs, scale_residual=True, prefix=None):
    shortcut = inputs
    
    branch1 = convolution_block(inputs, 192, (1, 1), name=f"inceptionC_step{prefix}_b12")

    branch2 = convolution_block(inputs, 192, (1, 1), name=f"inceptionC_step{prefix}_b21")
    branch2 = convolution_block(branch2, 224, (1, 3), name=f"inceptionC_step{prefix}_b22")
    branch2 = convolution_block(branch2, 256, (3, 1), name=f"inceptionC_step{prefix}_b23")

    x = concatenate([branch1, branch2], axis=-1, name=f'inceptionC_step{prefix}_submerged')
    x = convolution_block(x, 2144, (1, 1), activation=None, name=f'inceptionC_step{prefix}_linear')
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f'inceptionC_step{prefix}_scale')(x)
        
    out = add([shortcut, x], name=f'inceptionC_step{prefix}_merged')
    out = BatchNormalization(axis=-1, name=f'inceptionC_step{prefix}_bn')(out)
    out = Activation("relu", name=f'inceptionC_step{prefix}_activ')(out)
    return out


def Inception_Resnet_v2(depths=[1, 2, 1],
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # Stem block
    x = stem_block(img_input)

    # Inception-A
    for i in range(depths[0]):
        x = inception_resnet_A(x, scale_residual=scale_residual, prefix=str(i+1))

    # Reduction-A
    x = reduction_A(x, k=256, l=256, m=384, n=384)

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
        x = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(x)
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
    model = Model(inputs, x, name='Inception-Resnet-v2')

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
