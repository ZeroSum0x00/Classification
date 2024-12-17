"""
  # Description:
    - The following table comparing the params of the VGGNet in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |       VGG-11        |   132,863,336   |
      |---------------------|-----------------|
      |       VGG-13        |   133,047,848   |
      |---------------------|-----------------|
      |       VGG-16        |   138,357,544   |
      |---------------------|-----------------|
      |       VGG-19        |   143,667,240   |
       ---------------------------------------

  # Reference:
    - [Very deep convolutional networks for large-scale image 
       recognition](https://arxiv.org/pdf/1409.1556.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
              https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import get_activation_from_name
from utils.model_processing import _obtain_input_shape


BASE_WEIGTHS_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/')
VGG16_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
VGG16_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
VGG19_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
VGG19_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGGBlock(x, num_layers, filters, activation='relu', name='vgg_block'):
    for i in range(num_layers):
        x = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   name=name + "_conv" + str(i))(x)
        x = get_activation_from_name(activation, name=name + "_activ" + str(i))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name + "_pool")(x)
    return x


def VGG(layers,
        filters,
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

    # Block 0
    x = VGGBlock(img_input, layers[0], filters[0], 'relu', name='vgg_block0')

    # Block 1
    x = VGGBlock(x, layers[1], filters[1], 'relu', name='vgg_block1')

    # Block 2
    x = VGGBlock(x, layers[2], filters[2], 'relu', name='vgg_block2')

    # Block 3
    x = VGGBlock(x, layers[3], filters[3], 'relu', name='vgg_block3')

    # Block 4
    x = VGGBlock(x, layers[4], filters[4], 'relu', name='vgg_block4')

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc1')(x)
        x = get_activation_from_name('relu')(x)
        x = Dense(4096, name='fc2')(x)
        x = get_activation_from_name('relu')(x)
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
    if layers == [1, 1, 2, 2, 2]:
        model = Model(inputs, x, name='VGG-11')
    elif layers == [2, 2, 2, 2, 2]:
        model = Model(inputs, x, name='VGG-13')
    elif layers == [2, 2, 3, 3, 3]:
        model = Model(inputs, x, name='VGG-16')
    elif layers == [2, 2, 4, 4, 4]:
        model = Model(inputs, x, name='VGG-19')
    else:
        model = Model(inputs, x, name='VGG')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if layers == [1, 1, 2, 2, 2]:
                weights_path = None
            elif layers == [2, 2, 2, 2, 2]:
                weights_path = None
            elif layers == [2, 2, 3, 3, 3]:
                weights_path = get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                    VGG16_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
            elif layers == [2, 2, 4, 4, 4]:
                weights_path = get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                    VGG19_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            if layers == [1, 1, 2, 2, 2]:
                weights_path = None
            elif layers == [2, 2, 2, 2, 2]:
                weights_path = None
            elif layers == [2, 2, 3, 3, 3]:
                weights_path = get_file(
                    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    VGG16_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='6d6bbae143d832006294945121d1f1fc')
            elif layers == [2, 2, 4, 4, 4]:
                weights_path = get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    VGG19_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='253f8cb515780f3b799900260a226db6')
        if weights_path is not None:
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


def VGG11(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000) -> Model:
    
    model = VGG(layers=[1, 1, 2, 2, 2],
                filters=[64, 128, 256, 512, 512],
                include_top=include_top,
                weights=weights, 
                input_tensor=input_tensor, 
                input_shape=input_shape, 
                pooling=pooling, 
                final_activation=final_activation,
                classes=classes)
    return model


def VGG11_backbone(input_shape=(224, 224, 3), 
                   include_top=False, 
                   weights='imagenet', 
                   input_tensor=None, 
                   pooling=None, 
                   final_activation="softmax",
                   classes=1000,
                   custom_layers=None) -> Model:
    
    model = VGG11(include_top=include_top, 
                  weights=weights,
                  input_tensor=input_tensor, 
                  input_shape=input_shape,
                  pooling=pooling, 
                  final_activation=final_activation, 
                  classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='VGG11_backbone')

    else:
        y_2 = model.get_layer("vgg_block1_activ0").output
        y_4 = model.get_layer("vgg_block2_activ1").output
        y_8 = model.get_layer("vgg_block3_activ1").output
        y_16 = model.get_layer("vgg_block4_activ1").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='VGG11_backbone')


def VGG13(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000) -> Model:
    
    model = VGG(layers=[2, 2, 2, 2, 2],
                filters=[64, 128, 256, 512, 512],
                include_top=include_top,
                weights=weights, 
                input_tensor=input_tensor, 
                input_shape=input_shape, 
                pooling=pooling, 
                final_activation=final_activation,
                classes=classes)
    return model


def VGG13_backbone(input_shape=(224, 224, 3), 
                   include_top=False, 
                   weights='imagenet', 
                   input_tensor=None, 
                   pooling=None, 
                   final_activation="softmax",
                   classes=1000,
                   custom_layers=None) -> Model:
    
    model = VGG13(include_top=include_top, 
                  weights=weights,
                  input_tensor=input_tensor, 
                  input_shape=input_shape,
                  pooling=pooling, 
                  final_activation=final_activation, 
                  classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='VGG13_backbone')

    else:
        y_2 = model.get_layer("vgg_block1_activ1").output
        y_4 = model.get_layer("vgg_block2_activ1").output
        y_8 = model.get_layer("vgg_block3_activ1").output
        y_16 = model.get_layer("vgg_block4_activ1").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='VGG13_backbone')


def VGG16(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000) -> Model:
    
    model = VGG(layers=[2, 2, 3, 3, 3],
                filters=[64, 128, 256, 512, 512],
                include_top=include_top,
                weights=weights, 
                input_tensor=input_tensor, 
                input_shape=input_shape, 
                pooling=pooling, 
                final_activation=final_activation,
                classes=classes)
    return model


def VGG16_backbone(input_shape=(224, 224, 3), 
                   include_top=False, 
                   weights='imagenet', 
                   input_tensor=None, 
                   pooling=None, 
                   final_activation="softmax",
                   classes=1000,
                   custom_layers=None) -> Model:
    
    model = VGG16(include_top=include_top, 
                  weights=weights,
                  input_tensor=input_tensor, 
                  input_shape=input_shape,
                  pooling=pooling, 
                  final_activation=final_activation, 
                  classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='VGG16_backbone')

    else:
        y_2 = model.get_layer("vgg_block1_activ1").output
        y_4 = model.get_layer("vgg_block2_activ2").output
        y_8 = model.get_layer("vgg_block3_activ2").output
        y_16 = model.get_layer("vgg_block4_activ2").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='VGG16_backbone')


def VGG19(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000) -> Model:
    
    model = VGG(layers=[2, 2, 4, 4, 4],
                filters=[64, 128, 256, 512, 512],
                include_top=include_top,
                weights=weights, 
                input_tensor=input_tensor, 
                input_shape=input_shape, 
                pooling=pooling, 
                final_activation=final_activation,
                classes=classes)
    return model


def VGG19_backbone(input_shape=(224, 224, 3), 
                   include_top=False, 
                   weights='imagenet', 
                   input_tensor=None, 
                   pooling=None, 
                   final_activation="softmax",
                   classes=1000,
                   custom_layers=None) -> Model:
    
    model = VGG19(include_top=include_top, 
                  weights=weights,
                  input_tensor=input_tensor, 
                  input_shape=input_shape,
                  pooling=pooling, 
                  final_activation=final_activation, 
                  classes=classes)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name='VGG19_backbone')

    else:
        y_2 = model.get_layer("vgg_block1_activ1").output
        y_4 = model.get_layer("vgg_block2_activ3").output
        y_8 = model.get_layer("vgg_block3_activ3").output
        y_16 = model.get_layer("vgg_block4_activ3").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_final], name='VGG19_backbone')
