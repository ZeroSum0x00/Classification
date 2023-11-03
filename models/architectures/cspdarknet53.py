import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D

from .darknet53 import convolutional_block, residual_block
from models.layers import get_activation_from_name
from utils.model_processing import _obtain_input_shape


def CSPDarkNetBlock(inputs, num_filters, block_iter, activation='mish', norm_layer='batch-norm', name=None):
    route = convolutional_block(inputs, num_filters[1], 1, activation=activation, norm_layer=norm_layer, name=name + '_shortcut')
    
    x = convolutional_block(inputs, num_filters[1], 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')

    for i in range(block_iter):
        x = residual_block(x,  [num_filters[0], num_filters[1]], activation=activation, norm_layer=norm_layer, name=name + f'_residual{i + 1}')

    x = convolutional_block(x, num_filters[1], 1, activation=activation, norm_layer=norm_layer, name=name + '_conv2')
    x = concatenate([x, route], axis=-1, name=name + '_merger')
    x = convolutional_block(x, num_filters[0]*2, 1, activation=activation, norm_layer=norm_layer, name=name + '_projection')
    return x


def CSPDarkNet53(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 activation='mish',
                 norm_layer='batch-norm',
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
                                      default_size=640,
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

    x = convolutional_block(img_input, 32, 3, activation=activation, norm_layer=norm_layer, name="stem")

    # Downsample 1
    x = convolutional_block(x, 64, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage1_block1")
    
    # CSPResBlock 1
    x = CSPDarkNetBlock(x, [32, 64], 1, activation=activation, norm_layer=norm_layer, name="stage1_block2")

    # Downsample 2
    x = convolutional_block(x, 128, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage2_block1")

    # CSPResBlock 2
    x = CSPDarkNetBlock(x, [64, 64], 2, activation=activation, norm_layer=norm_layer, name="stage2_block2")

    # Downsample 3
    x = convolutional_block(x, 256, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage3_block1")

    # CSPResBlock 3
    x = CSPDarkNetBlock(x, [128, 128], 8, activation=activation, norm_layer=norm_layer, name="stage3_block2")

    # Downsample 4
    x = convolutional_block(x, 512, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage4_block1")

    # CSPResBlock 4
    x = CSPDarkNetBlock(x, [256, 256], 8, activation=activation, norm_layer=norm_layer, name="stage4_block2")

    # Downsample 5
    x = convolutional_block(x, 1024, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage5_block1")

    # CSPResBlock 5
    x = CSPDarkNetBlock(x, [512, 512], 4, activation=activation, norm_layer=norm_layer, name="stage5_block2")

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
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

    model = Model(inputs=inputs, outputs=x, name="CSPDarkNet-53")

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


def CSPDarkNet53_backbone(input_shape=(416, 416, 3),
                          include_top=False, 
                          weights='imagenet', 
                          activation='mish',
                          norm_layer='batch-norm',
                          custom_layers=None) -> Model:

    model = CSPDarkNet53(include_top=include_top, 
                         weights=weights,
                         activation=activation,
                         norm_layer=norm_layer,
                         input_shape=input_shape)

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=[y_i], name=model.name + '_backbone')
    else:
        y_2 = model.get_layer("stage1_block2_projection/activ").output
        y_4 = model.get_layer("stage2_block2_projection/activ").output
        y_8 = model.get_layer("stage3_block2_projection/activ").output
        y_16 = model.get_layer("stage4_block2_projection/activ").output
        y_32 = model.get_layer("stage5_block2_projection/activ").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')