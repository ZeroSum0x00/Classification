"""
  # Description:
    - On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
    and a top-5 validation accuracy of 0.945.
    - Also do note that this model is only available for the TensorFlow backend,
    due to its reliance on `SeparableConvolution` layers.
    - The following table comparing the params of the Extreme Inception (Xception) in 
    Tensorflow on size 299 x 299 x 3:

       --------------------------------------
      |     Model Name      |    Params      |
      |--------------------------------------|
      |     Xception        |   22,910,480   |
       --------------------------------------

  # Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from .utils import _obtain_input_shape

TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, 
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
                                      default_size=299,
                                      min_size=71,
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

    """ 
        Entry flow
    """
    # Block 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='block1_conv1_batchnorm')(x)
    x = Activation('relu', name='block1_conv1_activation')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block1_conv2_batchnorm')(x)
    x = Activation('relu', name='block1_conv2_activation')(x)

    residual = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_residual1')(x)
    residual = BatchNormalization(axis=bn_axis, name='entry_residual1_batchnorm')(residual)


    # Block 2
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv1_batchnorm')(x)

    x = Activation('relu', name='block2_sepconv1_activation')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block2_sepconv2_batchnorm')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block2_maxpooling')(x)

    x = add([x, residual], name='block2_merge')


    # Block 3
    residual = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_residual2')(x)
    residual = BatchNormalization(axis=bn_axis, name='entry_residual2_batchnorm')(residual)

    x = Activation('relu', name='block3_sepconv1_activation')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv1_batchnorm')(x)

    x = Activation('relu', name='block3_sepconv2_activation')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block3_sepconv2_batchnorm')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block3_maxpooling')(x)

    x = add([x, residual], name='block3_merge')


    # Block4
    residual = Conv2D(filters=728, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='entry_residual3')(x)
    residual = BatchNormalization(axis=bn_axis, name='entry_residual3_batchnorm')(residual)

    x = Activation('relu', name='block4_sepconv1_activation')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv1_batchnorm')(x)

    x = Activation('relu', name='block4_sepconv2_activation')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block4_sepconv2_batchnorm')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block4_maxpooling')(x)

    x = add([x, residual], name='block4_merge')


    """ 
        Middle flow
    """ 
    # Block 5 - 12
    for i in range(8):
        prefix = 'block' + str(i + 5)
        residual = x

        x = Activation('relu', name=prefix + '_sepconv1_activation')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv1_batchnorm')(x)

        x = Activation('relu', name=prefix + '_sepconv2_activation')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv2_batchnorm')(x)

        x = Activation('relu', name=prefix + '_sepconv3_activation')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=bn_axis, name=prefix + '_sepconv3_batchnorm')(x)

        x = add([x, residual], name='block' + str(i + 5) + '_merge')


    """ 
        Exit flow
    """ 
    # Block 13
    residual = Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=False, name='exit_residual1')(x)
    residual = BatchNormalization(axis=bn_axis, name='exit_residual1_batchnorm')(residual)

    x = Activation('relu', name='block13_sepconv1_activation')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv1_batchnorm')(x)

    x = Activation('relu', name='block13_sepconv2_activation')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block13_sepconv2_batchnorm')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='block13_maxpooling')(x)

    x = add([x, residual], name='block13_merge')


    # Block 14
    x = SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv1_batchnorm')(x)
    x = Activation('relu', name='block14_sepconv1_activation')(x)

    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(axis=bn_axis, name='block14_sepconv2_batchnorm')(x)
    x = Activation('relu', name='block14_sepconv2_activation')(x)

    # Final Block
    if include_top:
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        x = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avgpool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_maxpool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x, name='Xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path)

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


def Xception_backbone(input_shape=(299, 299, 3),
                      include_top=False, 
                      weights='imagenet', 
                      input_tensor=None, 
                      pooling=None, 
                      final_activation="softmax",
                      classes=1000,
                      custom_layers=None) -> Model:

    model = Xception(include_top=include_top, weights=weights,
                     input_tensor=input_tensor, input_shape=input_shape,
                     pooling=pooling, final_activation=final_activation, classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='Xception_backbone')

    else:
        y_2 = model.get_layer("block2_sepconv2_batchnorm").output
        y_4 = model.get_layer("block3_sepconv2_batchnorm").output
        y_8 = model.get_layer("block4_sepconv2_batchnorm").output
        y_16 = model.get_layer("block13_sepconv2_batchnorm").output
        y_32 = model.get_layer("block14_sepconv2_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='Xception_backbone')

def Xception_backbone2(input_shape=(299, 299, 3), 
                       include_top=False, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       final_activation="softmax",
                       classes=1000,
                       custom_layers=None) -> Model:
    
    model = Xception(include_top=include_top, weights=weights,
                     input_tensor=input_tensor, input_shape=input_shape,
                     pooling=pooling, final_activation=final_activation, classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='Xception_backbone')

    else:
        y_2 = model.get_layer("block2_merge").output
        y_4 = model.get_layer("block3_merge").output
        y_8 = model.get_layer("block4_merge").output
        y_16 = model.get_layer("block12_merge").output
        y_32 = model.get_layer("block14_sepconv2_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='Xception_backbone')


if __name__ == '__main__':
    model = Xception(input_shape=(299, 299, 3), include_top=False, pooling='avg')
