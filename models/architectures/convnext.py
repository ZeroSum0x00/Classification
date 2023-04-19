"""
  # Description:
    - The following table comparing the params of the ConvNeXt in Pytorch Source 
    with Tensorflow convert Source on size 224 x 224 x 3:
      
       ----------------------------------------------------------------------
      |    Library     |     Model Name      |    Params       |   Greater   |
      |--------------------------------------------------------|-------------|
      |   Pytorch      |     ConvNeXt-T      |   28,582,504    |      =      |
      |   Tensorflow   |     ConvNeXt-T      |   28,582,504    |      =      |
      |----------------|---------------------|-----------------|-------------|
      |   Pytorch      |     ConvNeXt-S      |   50,210,152    |      =      |
      |   Tensorflow   |     ConvNeXt-S      |   50,210,152    |      =      |
      |----------------|---------------------|-----------------|-------------|
      |   Pytorch      |     ConvNeXt-B      |   88,573,416    |      =      |
      |   Tensorflow   |     ConvNeXt-B      |   88,573,416    |      =      |
      |----------------|---------------------|-----------------|-------------|
      |   Pytorch      |     ConvNeXt-L      |   197,740,264   |      =      |
      |   Tensorflow   |     ConvNeXt-L      |   197,740,264   |      =      |
      |----------------|---------------------|-----------------|-------------|
      |   Pytorch      |     ConvNeXt-XL     |   350,160,872   |      =      |
      |   Tensorflow   |     ConvNeXt-XL     |   350,160,872   |      =      |
       ----------------------------------------------------------------------

  # Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf)
    - Source: https://github.com/facebookresearch/ConvNeXt

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras import backend as K
from .utils import _obtain_input_shape

try:
      from ..layers.stochastic_depth import StochasticDepth
except ImportError:
      from ..layers.stochastic_depth import StochasticDepth2 as StochasticDepth

kernel_initial = tf.keras.initializers.TruncatedNormal(stddev=0.2)
bias_initial = tf.keras.initializers.Constant(value=0)



def stem_cell(inputs, out_filter, norm_eps=1e-6, name='stem'):
    x = Conv2D(filters=out_filter, 
               kernel_size=(4, 4), 
               strides=(4, 4), 
               padding='same', 
               name=name + '_conv')(inputs)
    x = LayerNormalization(epsilon=norm_eps, name=name + '_norm')(x)
    return x


def Downsamples(inputs, 
                out_filter, 
                kernel_initial=kernel_initial, 
                bias_initial=bias_initial,
                norm_eps=1e-6, 
                name="downsamples"):
    x = LayerNormalization(epsilon=norm_eps, name=name + '_norm')(inputs)
    x = Conv2D(filters=out_filter,
               kernel_size=(2, 2), 
               strides=(2, 2), 
               padding='same',
               kernel_initializer=kernel_initial,
               bias_initializer=bias_initial, 
               name=name + '_conv')(x)
    return x


def ConvNextBlock(inputs, 
                  drop_prob=0, 
                  layer_scale_init_value=1e-6, 
                  kernel_initial=kernel_initial, 
                  bias_initial=bias_initial,
                  norm_eps=1e-6,
                  name="block"):
  
    in_filters = inputs.shape[-1]

    x = Conv2D(filters=in_filters,
               kernel_size=(7, 7),
               strides=(1, 1),
               padding="same",
               groups=in_filters,
               kernel_initializer=kernel_initial,
               bias_initializer=bias_initial,
               name=name + "_dw")(inputs)
    x = LayerNormalization(epsilon=norm_eps, name=name + '_norm')(x)
    x = Conv2D(filters=in_filters*4,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               kernel_initializer=kernel_initial,
               bias_initializer=bias_initial,
               name=name + "_pw")(x)
    x = Activation('gelu', name=name + '_activation')(x)
    x = Conv2D(filters=in_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               kernel_initializer=kernel_initial,
               bias_initializer=bias_initial,
               name=name + "_conv_final")(x)

    if layer_scale_init_value > 0:
        layer_scale_gamma = tf.ones(in_filters) * layer_scale_init_value
        # layer_scale_gamma = tf.Variable(initial_value=layer_scale_init_value*)
        x = x * layer_scale_gamma

    if drop_prob > 0:
        x = StochasticDepth(drop_prob, name=name + "_droppath")([inputs, x])
    
    x = Lambda(lambda x: x, name=name + "_final")(x)
    return x


def ConvNext(depths=[3, 3, 9, 3], 
             dims=[96, 192, 384, 768], 
             include_top=True, 
             weights='imagenet',
             input_tensor=None, 
             input_shape=None,
             pooling=None,
             classes=1000,
             drop_path_rate=0., 
             layer_scale_init_value=1e-6,
             kernel_initial=kernel_initial,
             norm_eps=1e-6,
             bias_initial=bias_initial):

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
    
    cur = 0
    dp_rates = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

    x = stem_cell(img_input, dims[0], norm_eps)
    for i in range(depths[0]):
        x = ConvNextBlock(x, 
                          dp_rates[cur + i],
                          layer_scale_init_value,
                          kernel_initial,
                          bias_initial,
                          norm_eps,
                          name='block0_part' + str(i))
    
    cur += depths[0]
    x = Downsamples(x, dims[1], kernel_initial, bias_initial, norm_eps, name='downsaples1')
    for i in range(depths[1]):
        x = ConvNextBlock(x, 
                          dp_rates[cur + i], 
                          layer_scale_init_value,
                          kernel_initial,
                          bias_initial,
                          norm_eps,
                          name='block1_part' + str(i))

    cur += depths[1]
    x = Downsamples(x, dims[2], kernel_initial, bias_initial, norm_eps, name='downsaples2')
    for i in range(depths[2]):
        x = ConvNextBlock(x, 
                          dp_rates[cur + i], 
                          layer_scale_init_value,
                          kernel_initial,
                          bias_initial,
                          norm_eps,
                          name='block2_part' + str(i))

    cur += depths[2]
    x = Downsamples(x, dims[3], kernel_initial, bias_initial, norm_eps, name='downsaples3')
    for i in range(depths[3]):
        x = ConvNextBlock(x, 
                          dp_rates[cur + i], 
                          layer_scale_init_value,
                          kernel_initial,
                          bias_initial,
                          norm_eps,
                          name='block3_part' + str(i))

    if include_top:
        x = GlobalAveragePooling2D(name='global_avgpool')(x)
        x = LayerNormalization(epsilon=norm_eps, name='final_norm')(x)
        x = Dense(classes, 
                  activation='softmax',
                  name='predictions',
                  kernel_initializer=kernel_initial, 
                  bias_initializer=bias_initial)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avgpool')(x)
            x = LayerNormalization(epsilon=norm_eps, name='final_norm')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_maxpool')(x)
            x = LayerNormalization(epsilon=norm_eps, name='final_norm')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    if depths == [3, 3, 9, 3] and dims == [96, 192, 384, 768]:
        model = Model(inputs, x, name='ConvNext-T')
    elif depths == [3, 3, 27, 3] and dims == [96, 192, 384, 768]:
        model = Model(inputs, x, name='ConvNext-S')
    elif depths == [3, 3, 27, 3] and dims == [128, 256, 512, 1024]:
        model = Model(inputs, x, name='ConvNext-B')
    elif depths == [3, 3, 27, 3] and dims == [192, 384, 768, 1536]:
        model = Model(inputs, x, name='ConvNext-L')
    elif depths == [3, 3, 27, 3] and dims == [256, 512, 1024, 2048]:
        model = Model(inputs, x, name='ConvNext-XL')
    else:
        model = Model(inputs, x, name='ConvNext')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if depths == [3, 3, 9, 3] and dims == [96, 192, 384, 768]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [96, 192, 384, 768]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [128, 256, 512, 1024]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [192, 384, 768, 1536]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [256, 512, 1024, 2048]:
                weights_path = None
        else:
            if depths == [3, 3, 9, 3] and dims == [96, 192, 384, 768]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [96, 192, 384, 768]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [128, 256, 512, 1024]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [192, 384, 768, 1536]:
                weights_path = None
            elif depths == [3, 3, 27, 3] and dims == [256, 512, 1024, 2048]:
                weights_path = None
        if weights_path is not None:
            model.load_weights(weights_path)
    
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
   

def ConvNextT(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              drop_path_rate=0., 
              layer_scale_init_value=1e-6,
              kernel_initial=kernel_initial, 
              bias_initial=bias_initial,
              norm_eps=1e-6) -> Model:
    
    model = ConvNext(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768], 
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     classes=classes,
                     drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     kernel_initial=kernel_initial,
                     bias_initial=bias_initial,
                     norm_eps=norm_eps)
    return model


def ConvNextT_backbone(input_shape=(224, 224, 3), 
                       include_top=True, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       classes=1000,
                       drop_path_rate=0., 
                       layer_scale_init_value=1e-6,
                       kernel_initial=kernel_initial, 
                       bias_initial=bias_initial,
                       norm_eps=1e-6,
                       custom_layers=None) -> Model:

    model = ConvNextT(include_top=include_top, 
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      classes=classes, 
                      drop_path_rate=drop_path_rate,
                      layer_scale_init_value=layer_scale_init_value,
                      kernel_initial=kernel_initial, 
                      bias_initial=bias_initial,
                      norm_eps=norm_eps)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='ConvNextT_backbone')

    else:
        y_4 = model.get_layer("block0_part2_final").output
        y_8 = model.get_layer("block1_part2_final").output
        y_16 = model.get_layer("block2_part8_final").output
        y_32 = model.get_layer("block3_part2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_4, y_8, y_16, y_32, y_final], name='ConvNextT_backbone')


def ConvNextS(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              drop_path_rate=0., 
              layer_scale_init_value=1e-6,
              kernel_initial=kernel_initial, 
              bias_initial=bias_initial,
              norm_eps=1e-6) -> Model:
    
    model = ConvNext(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768], 
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     classes=classes,
                     drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     kernel_initial=kernel_initial,
                     bias_initial=bias_initial,
                     norm_eps=norm_eps)
    return model


def ConvNextS_backbone(input_shape=(224, 224, 3), 
                       include_top=True, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       classes=1000,
                       drop_path_rate=0., 
                       layer_scale_init_value=1e-6,
                       kernel_initial=kernel_initial, 
                       bias_initial=bias_initial,
                       norm_eps=1e-6,
                       custom_layers=None) -> Model:

    model = ConvNextS(include_top=include_top, 
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      classes=classes, 
                      drop_path_rate=drop_path_rate, 
                      layer_scale_init_value=layer_scale_init_value,
                      kernel_initial=kernel_initial, 
                      bias_initial=bias_initial,
                      norm_eps=norm_eps)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='ConvNextS_backbone')

    else:
        y_4 = model.get_layer("block0_part2_final").output
        y_8 = model.get_layer("block1_part2_final").output
        y_16 = model.get_layer("block2_part26_final").output
        y_32 = model.get_layer("block3_part2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_4, y_8, y_16, y_32, y_final], name='ConvNextS_backbone')


def ConvNextB(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              drop_path_rate=0., 
              layer_scale_init_value=1e-6,
              kernel_initial=kernel_initial, 
              bias_initial=bias_initial,
              norm_eps=1e-6) -> Model:
    
    model = ConvNext(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024], 
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     classes=classes,
                     drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     kernel_initial=kernel_initial,
                     bias_initial=bias_initial,
                     norm_eps=norm_eps)
    return model


def ConvNextB_backbone(input_shape=(224, 224, 3), 
                       include_top=True, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       classes=1000,
                       drop_path_rate=0., 
                       layer_scale_init_value=1e-6,
                       kernel_initial=kernel_initial, 
                       bias_initial=bias_initial,
                       norm_eps=1e-6,
                       custom_layers=None) -> Model:

    model = ConvNextB(include_top=include_top, 
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      classes=classes, 
                      drop_path_rate=drop_path_rate, 
                      layer_scale_init_value=layer_scale_init_value,
                      kernel_initial=kernel_initial, 
                      bias_initial=bias_initial,
                      norm_eps=norm_eps)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='ConvNextB_backbone')

    else:
        y_4 = model.get_layer("block0_part2_final").output
        y_8 = model.get_layer("block1_part2_final").output
        y_16 = model.get_layer("block2_part26_final").output
        y_32 = model.get_layer("block3_part2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_4, y_8, y_16, y_32, y_final], name='ConvNextB_backbone')


def ConvNextL(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              drop_path_rate=0., 
              layer_scale_init_value=1e-6,
              kernel_initial=kernel_initial, 
              bias_initial=bias_initial,
              norm_eps=1e-6) -> Model:
    
    model = ConvNext(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536], 
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     classes=classes,
                     drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     kernel_initial=kernel_initial,
                     bias_initial=bias_initial,
                     norm_eps=norm_eps)
    return model


def ConvNextL_backbone(input_shape=(224, 224, 3), 
                       include_top=True, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       classes=1000,
                       drop_path_rate=0., 
                       layer_scale_init_value=1e-6,
                       kernel_initial=kernel_initial, 
                       bias_initial=bias_initial,
                       norm_eps=1e-6,
                       custom_layers=None) -> Model:

    model = ConvNextL(include_top=include_top, 
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      classes=classes, 
                      drop_path_rate=drop_path_rate, 
                      layer_scale_init_value=layer_scale_init_value,
                      kernel_initial=kernel_initial, 
                      bias_initial=bias_initial,
                      norm_eps=norm_eps)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='ConvNextL_backbone')

    else:
        y_4 = model.get_layer("block0_part2_final").output
        y_8 = model.get_layer("block1_part2_final").output
        y_16 = model.get_layer("block2_part26_final").output
        y_32 = model.get_layer("block3_part2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_4, y_8, y_16, y_32, y_final], name='ConvNextL_backbone')


def ConvNextXL(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              drop_path_rate=0., 
              layer_scale_init_value=1e-6,
              kernel_initial=kernel_initial, 
              bias_initial=bias_initial,
              norm_eps=1e-6) -> Model:
    
    model = ConvNext(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048], 
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     classes=classes,
                     drop_path_rate=drop_path_rate,
                     layer_scale_init_value=layer_scale_init_value,
                     kernel_initial=kernel_initial,
                     bias_initial=bias_initial,
                     norm_eps=norm_eps)
    return model


def ConvNextXL_backbone(input_shape=(224, 224, 3), 
                       include_top=True, 
                       weights='imagenet', 
                       input_tensor=None, 
                       pooling=None, 
                       classes=1000,
                       drop_path_rate=0., 
                       layer_scale_init_value=1e-6,
                       kernel_initial=kernel_initial, 
                       bias_initial=bias_initial,
                       norm_eps=1e-6,
                       custom_layers=None) -> Model:

    model = ConvNextXL(include_top=include_top, 
                      weights=weights,
                      input_tensor=input_tensor, 
                      input_shape=input_shape,
                      pooling=pooling, 
                      classes=classes, 
                      drop_path_rate=drop_path_rate, 
                      layer_scale_init_value=layer_scale_init_value,
                      kernel_initial=kernel_initial, 
                      bias_initial=bias_initial,
                      norm_eps=norm_eps)

    for l in model.layers:
        l.trainable = True

    if custom_layers is not None:
        y_i = []
        for layer in custom_layers:
            y_i.append(model.get_layer(layer).output)
        return Model(inputs=model.inputs, outputs=y_i, name='ConvNextXL_backbone')

    else:
        y_4 = model.get_layer("block0_part2_final").output
        y_8 = model.get_layer("block1_part2_final").output
        y_16 = model.get_layer("block2_part26_final").output
        y_32 = model.get_layer("block3_part2_final").output
        y_final = model.get_layer(model.layers[-1].name).output
        return Model(inputs=model.inputs, outputs=[y_4, y_8, y_16, y_32, y_final], name='ConvNextXL_backbone')


if __name__ == "__main__":
    model = ConvNextL(input_shape=(224, 224, 3), weights=None)
    model.summary()