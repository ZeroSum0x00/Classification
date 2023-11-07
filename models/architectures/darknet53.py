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


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size       = 3, 
                 downsample        = False, 
                 dilation_rate     = (1, 1),
                 groups            = 1,
                 activation        = 'leaky', 
                 norm_layer        = 'batch-norm', 
                 regularizer_decay = 5e-4,
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters           = filters
        self.kernel_size       = kernel_size
        self.downsample        = downsample
        self.dilation_rate     = dilation_rate
        self.groups            = groups
        self.activation        = activation
        self.norm_layer        = norm_layer
        self.regularizer_decay = regularizer_decay
        
        if downsample:
            self.padding = 'valid'
            self.strides = 2
        else:
            self.padding = 'same'
            self.strides = 1

    def build(self, input_shape):
        self.padding_layer = ZeroPadding2D(padding=((1, 0), (1, 0))) if self.downsample else None
        self.conv = Conv2D(filters=self.filters, 
                           kernel_size=self.kernel_size, 
                           strides=self.strides,
                           padding=self.padding, 
                           dilation_rate=self.dilation_rate,
                           groups=self.groups,
                           use_bias=False if self.norm_layer else True, 
                           kernel_initializer=RandomNormal(stddev=0.02),
                           kernel_regularizer=l2(self.regularizer_decay))
        self.norm_layer = get_normalizer_from_name(self.norm_layer)
        self.activation = get_activation_from_name(self.activation)

    def call(self, inputs, training=False):
        if self.downsample:
            inputs = self.padding_layer(inputs)
        x = self.conv(inputs, training=training)
        if self.norm_layer:
            x = self.norm_layer(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 num_filters, 
                 activation        = 'leaky', 
                 norm_layer        = 'batch-norm', 
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        self.conv1 = ConvolutionBlock(self.num_filters[0], 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv2 = ConvolutionBlock(self.num_filters[1], 3, activation=self.activation, norm_layer=self.norm_layer)
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return add([inputs, x])


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
        
    x = ConvolutionBlock(32, 3, activation=activation, norm_layer=norm_layer, name="stage1_block1")(img_input)
    
    x = ConvolutionBlock(64, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage2_block1")(x)
    
    for i in range(1):
        x = ResidualBlock(num_filters=[32, 64], activation=activation, norm_layer=norm_layer, name=f'stage2_block{i + 2}')(x)

    x = ConvolutionBlock(128, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage3_block1")(x)

    for i in range(2):
        x = ResidualBlock(num_filters=[64, 128], activation=activation, norm_layer=norm_layer, name=f'stage3_block{i + 2}')(x)

    x = ConvolutionBlock(256, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage4_block1")(x)

    for i in range(8):
        x = ResidualBlock(num_filters=[128, 256], activation=activation, norm_layer=norm_layer, name=f'stage4_block{i + 2}')(x)

    x = ConvolutionBlock(512, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage5_block1")(x)

    for i in range(8):
        x = ResidualBlock(num_filters=[256, 512], activation=activation, norm_layer=norm_layer, name=f'stage5_block{i + 2}')(x)

    x = ConvolutionBlock(1024, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage6_block1")(x)

    for i in range(4):
        x = ResidualBlock(num_filters=[512, 1024], activation=activation, norm_layer=norm_layer, name=f'stage6_block{i + 2}')(x)
        
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
        y_2 = model.get_layer("stage2_block2").output
        y_4 = model.get_layer("stage3_block3").output
        y_8 = model.get_layer("stage4_block9").output
        y_16 = model.get_layer("stage5_block9").output
        y_32 = model.get_layer("stage6_block5").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')