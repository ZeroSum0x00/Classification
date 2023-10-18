"""
  # Description:
    - The following table comparing the params of the DenseNet in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------
      |     Model Name      |    Params      |
      |--------------------------------------|
      |     DenseNet-121    |   8,062,504    |
      |---------------------|----------------|
      |     DenseNet-169    |   14,307,880   |
      |---------------------|----------------|
      |     DenseNet-201    |   20,242,984   |
      |---------------------|----------------|
      |     DenseNet-264    |   33,736,232   |
       --------------------------------------

  # Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
              https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import get_activation_from_name, get_nomalizer_from_name
from utils.model_processing import _obtain_input_shape


BASE_WEIGTHS_PATH = ('https://github.com/keras-team/keras-applications/releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv_block(inputs, growth_rate, activation='relu', normalizer='batch-norm', name=None):
    """
    A building block for a dense block

    :param inputs: input tensor.
    :param growth_rate: float, growth rate at dense layers.
    :param name: string, block label.
    :return: Output tensor for the block.
    """
    x = get_nomalizer_from_name(normalizer, epsilon=1.001e-5, name=name + '_bn0')(inputs)
    x = get_activation_from_name(activation, name=name + '_activation0')(x)
    x = Conv2D(filters=growth_rate * 4, kernel_size=(1, 1), strides=(1, 1), use_bias=False, name=name + '_conv1')(x)
    x = get_nomalizer_from_name(normalizer, epsilon=1.001e-5, name=name + '_bn1')(x)
    x = get_activation_from_name(activation, name=name + '_activation1')(x)
    x = Conv2D(filters=growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name=name + '_conv2')(x)
    merge = concatenate([inputs, x], name=name + '_merge')
    return merge


def dense_block(inputs, blocks, activation='relu', normalizer='batch-norm', name=None):
    """
    A dense block.

    :param inputs: input tensor.
    :param blocks: integer, the number of building blocks.
    :param name: string, block label.
    :return: output tensor for the block.
    """
    x = inputs
    for i in range(blocks):
        x = conv_block(x, growth_rate=32, activation=activation, normalizer=normalizer, name=name + '_block' + str(i + 1))
    return x


def transition_block(inputs, reduction, activation='relu', normalizer='batch-norm', name=None):
    """
    A transition block.

    :param inputs: input tensor.
    :param reduction: float, compression rate at transition layers.
    :param name: string, block label.
    :return: output tensor for the block.
    """
    x = get_nomalizer_from_name(normalizer, epsilon=1.001e-5, name=name + '_bn')(inputs)
    x = get_activation_from_name(activation, name=name + '_activation')(x)
    x = Conv2D(filters=int(K.int_shape(x)[bn_axis] * reduction), 
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=name + '_pool')(x)
    return x


def DenseNet(blocks,
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

    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='padding_0')(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), use_bias=False, name='conv1/conv')(x)
    x = get_nomalizer_from_name('batch-norm', epsilon=1.001e-5, name='conv1/bn')(x)
    x = get_activation_from_name('relu', name='conv1/activation')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    
    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = get_nomalizer_from_name('batch-norm', epsilon=1.001e-5, name='final_bn')(x)
    x = get_activation_from_name('relu', name='final_activation')(x)
                 
    if include_top:
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
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
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='DenseNet-121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='DenseNet-169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='DenseNet-201')
    elif blocks == [6, 12, 64, 48]:
        model = Model(inputs, x, name='DenseNet-264')
    else:
        model = Model(inputs, x, name='DenseNet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
            elif blocks == [6, 12, 64, 48]:
                weights_path = None
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
            elif blocks == [6, 12, 64, 48]:
                weights_path = None
                
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


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:
    
    model = DenseNet(blocks=[6, 12, 24, 16],
                     include_top=include_top, 
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def DenseNet121_backbone(input_shape=(224, 224, 3), 
                         include_top=True, 
                         weights='imagenet', 
                         input_tensor=None, 
                         pooling=None, 
                         final_activation="softmax",
                         classes=1000,
                         custom_layers=None) -> Model:
    
    model = DenseNet121(include_top=include_top, 
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
        return Model(inputs=model.inputs, outputs=[y_i], name='DenseNet121_backbone')

    else:
        y_2 = model.get_layer("pool1").output
        y_4 = model.get_layer("pool2_pool").output
        y_8 = model.get_layer("pool3_pool").output
        y_16 = model.get_layer("pool4_pool").output
        y_32 = model.get_layer("final_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='DenseNet121_backbone')


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:

    model = DenseNet(blocks=[6, 12, 32, 32],
                     include_top=include_top, 
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def DenseNet169_backbone(input_shape=(224, 224, 3), 
                         include_top=True, 
                         weights='imagenet', 
                         input_tensor=None, 
                         pooling=None, 
                         final_activation="softmax",
                         classes=1000,
                         custom_layers=None) -> Model:

    model = DenseNet169(include_top=include_top, 
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
        return Model(inputs=model.inputs, outputs=y_i, name='DenseNet169_backbone')

    else:
        y_2 = model.get_layer("pool1").output
        y_4 = model.get_layer("pool2_pool").output
        y_8 = model.get_layer("pool3_pool").output
        y_16 = model.get_layer("pool4_pool").output
        y_32 = model.get_layer("final_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='DenseNet169_backbone')


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:

    model = DenseNet(blocks=[6, 12, 48, 32],
                     include_top=include_top, 
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def DenseNet201_backbone(input_shape=(224, 224, 3), 
                         include_top=True, 
                         weights='imagenet', 
                         input_tensor=None, 
                         pooling=None, 
                         final_activation="softmax",
                         classes=1000,
                         custom_layers=None) -> Model:
    
    model = DenseNet201(include_top=include_top, 
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
        return Model(inputs=model.inputs, outputs=[y_i], name='DenseNet201_backbone')

    else:
        y_2 = model.get_layer("pool1").output
        y_4 = model.get_layer("pool2_pool").output
        y_8 = model.get_layer("pool3_pool").output
        y_16 = model.get_layer("pool4_pool").output
        y_32 = model.get_layer("final_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='DenseNet201_backbone')


def DenseNet264(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000) -> Model:

    model = DenseNet(blocks=[6, 12, 64, 48],
                     include_top=include_top, 
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes)
    return model


def DenseNet264_backbone(input_shape=(224, 224, 3), 
                         include_top=True, 
                         weights='imagenet', 
                         input_tensor=None, 
                         pooling=None, 
                         final_activation="softmax",
                         classes=1000,
                         custom_layers=None) -> Model:
    
    model = DenseNet264(include_top=include_top, 
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
        return Model(inputs=model.inputs, outputs=[y_i], name='DenseNet264_backbone')

    else:
        y_2 = model.get_layer("pool1").output
        y_4 = model.get_layer("pool2_pool").output
        y_8 = model.get_layer("pool3_pool").output
        y_16 = model.get_layer("pool4_pool").output
        y_32 = model.get_layer("final_activation").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32, y_final], name='DenseNet264_backbone')


if __name__ == '__main__':
    model = DenseNet121(input_shape=(224, 224, 3), include_top=False)