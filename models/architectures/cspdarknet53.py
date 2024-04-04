from __future__ import print_function
from __future__ import absolute_import

import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import get_source_inputs, get_file

from .darknet53 import ConvolutionBlock, ResidualBlock
from models.layers import get_activation_from_name
from utils.model_processing import _obtain_input_shape


class CSPDarkNetBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 num_filters, 
                 iters,
                 activation        = 'leaky', 
                 norm_layer        = 'batch-norm', 
                 **kwargs):
        super(CSPDarkNetBlock, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.iters       = iters
        self.activation  = activation
        self.norm_layer  = norm_layer
                     
    def build(self, input_shape):
        self.shortcut = ConvolutionBlock(self.num_filters[1], 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv1 = ConvolutionBlock(self.num_filters[1], 1, activation=self.activation, norm_layer=self.norm_layer)
        self.middle = Sequential([
            ResidualBlock([self.num_filters[0], self.num_filters[1]], activation=self.activation, norm_layer=self.norm_layer) for i in range(self.iters)
        ])
        self.conv2 = ConvolutionBlock(self.num_filters[1], 1, activation=self.activation, norm_layer=self.norm_layer)
        self.conv3 = ConvolutionBlock(self.num_filters[0] * 2, 1, activation=self.activation, norm_layer=self.norm_layer)
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.middle(x, training=training)
        x = self.conv2(x, training=training)
        y = self.shortcut(inputs, training=training)

        merger = concatenate([x, y], axis=-1)
        merger = self.conv3(merger, training=training)
        return merger

        
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

    x = ConvolutionBlock(32, 3, activation=activation, norm_layer=norm_layer, name="stem")(img_input)

    # Downsample 1
    x = ConvolutionBlock(64, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage1_block1")(x)
    
    # CSPResBlock 1
    x = CSPDarkNetBlock([32, 64], 1, activation=activation, norm_layer=norm_layer, name="stage1_block2")(x)

    # Downsample 2
    x = ConvolutionBlock(128, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage2_block1")(x)

    # CSPResBlock 2
    x = CSPDarkNetBlock([64, 64], 2, activation=activation, norm_layer=norm_layer, name="stage2_block2")(x)

    # Downsample 3
    x = ConvolutionBlock(256, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage3_block1")(x)

    # CSPResBlock 3
    x = CSPDarkNetBlock([128, 128], 8, activation=activation, norm_layer=norm_layer, name="stage3_block2")(x)

    # Downsample 4
    x = ConvolutionBlock(512, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage4_block1")(x)

    # CSPResBlock 4
    x = CSPDarkNetBlock([256, 256], 8, activation=activation, norm_layer=norm_layer, name="stage4_block2")(x)

    # Downsample 5
    x = ConvolutionBlock(1024, 3, downsample=True, activation=activation, norm_layer=norm_layer, name="stage5_block1")(x)

    # CSPResBlock 5
    x = CSPDarkNetBlock([512, 512], 4, activation=activation, norm_layer=norm_layer, name="stage5_block2")(x)

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
        y_2 = model.get_layer("stage1_block2").output
        y_4 = model.get_layer("stage2_block2").output
        y_8 = model.get_layer("stage3_block2").output
        y_16 = model.get_layer("stage4_block2").output
        y_32 = model.get_layer("stage5_block2").output
        return Model(inputs=model.inputs, outputs=[y_2, y_4, y_8, y_16, y_32], name=model.name + '_backbone')