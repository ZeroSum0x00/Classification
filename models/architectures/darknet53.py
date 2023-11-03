import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def convolutional_block(x, 
                        filters, 
                        kernel_size, 
                        dilation_rate=(1, 1), 
                        groups=1, 
                        downsample=False, 
                        activation='leaky', 
                        norm_layer='batch-norm', 
                        regularizer_decay=5e-4,
                        name=None):
    if downsample:
        x = ZeroPadding2D(padding=((1, 0), (1, 0)), name=name + "/padding")(x)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               strides=strides,
               padding=padding, 
               dilation_rate=dilation_rate,
               groups=groups,
               use_bias=not norm_layer, 
               kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(regularizer_decay), 
               name=name + "/conv")(x)
                            
    if norm_layer:
        x = get_normalizer_from_name(norm_layer, name=name + "/norm")(x)
        
    if activation:
        x = get_activation_from_name(activation, name=name + "/activ")(x)
        
    return x


def residual_block(x, num_filters, activation='leaky', norm_layer='batch-norm', name=None):
    shortcut = x
    x = convolutional_block(x, num_filters[0], 1, activation=activation, norm_layer=norm_layer, name=name + '_conv1')
    x = convolutional_block(x, num_filters[1], 3, activation=activation, norm_layer=norm_layer, name=name + '_conv2')
    x = add([shortcut, x], name=name + '_residual')
    return x


def DarkNet53(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              activation='leaky',
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
        
    x = convolutional_block(img_input, 32, 3, activation=activation, norm_layer=norm_layer, name="stage1_block1")
    
    x = convolutional_block(x, 64, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage2_block1")
    
    for i in range(1):
        x = residual_block(x,  [32, 64], activation=activation, norm_layer=norm_layer, name=f'stage2_block{i + 1}')

    x = convolutional_block(x, 128, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage3_block1")

    for i in range(2):
        x = residual_block(x, [64, 128], activation=activation, norm_layer=norm_layer, name=f'stage3_block{i + 1}')

    x = convolutional_block(x, 256, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage4_block1")

    for i in range(8):
        x = residual_block(x, [128, 256], activation=activation, norm_layer=norm_layer, name=f'stage4_block{i + 1}')

    x = convolutional_block(x, 512, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage5_block1")

    for i in range(8):
        x = residual_block(x, [256, 512], activation=activation, norm_layer=norm_layer, name=f'stage5_block{i + 1}')

    x = convolutional_block(x, 1024, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage6_block1")

    for i in range(4):
        x = residual_block(x, [512, 1024], activation=activation, norm_layer=norm_layer, name=f'stage6_block{i + 1}')
        
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

    model = Model(inputs, x, name='DarkNet-53')


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


def DarkNet53_backbone(input_shape=(416, 416, 3),
                       include_top=False, 
                       weights='imagenet', 
                       activation='leaky',
                       norm_layer='batch-norm',
                       custom_layers=None) -> Model:

    model = DarkNet53(include_top=include_top, 
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
        y_2 = model.get_layer("stage2_block1_residual").output
        y_4 = model.get_layer("stage3_block2_residual").output
        y_8 = model.get_layer("stage4_block8_residual").output
        y_16 = model.get_layer("stage5_block8_residual").output
        y_32 = model.get_layer("stage6_block4_residual").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')