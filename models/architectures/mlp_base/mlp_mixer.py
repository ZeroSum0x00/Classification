"""
  # Description:
    - The following table comparing the params of the MLP-Mixer in Pytorch Source 
    with Tensorflow convert Source on size 224 x 224 x 3:
      
       ---------------------------------------------------------------------------
      |    Library     |        Model Name        |    Params       |   Greater   |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-small-16    |   18,528,264    |      =      |
      |   Tensorflow   |     MLPMixer-small-16    |   18,528,264    |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-small-32    |   19,104,624    |      =      |
      |   Tensorflow   |     MLPMixer-small-32    |   19,104,624    |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-base-16     |   59,880,472    |      =      |
      |   Tensorflow   |     MLPMixer-base-16     |   59,880,472    |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-base-32     |   60,293,428    |      =      |
      |   Tensorflow   |     MLPMixer-base-32     |   60,293,428    |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-large-16    |   208,196,168   |      =      |
      |   Tensorflow   |     MLPMixer-large-16    |   208,196,168   |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-large-32    |   206,939,264   |      =      |
      |   Tensorflow   |     MLPMixer-large-32    |   206,939,264   |      =      |
      |---------------------------------------------------------------------------|
      |   Pytorch      |     MLPMixer-huge-14     |   432,350,952   |      =      |
      |   Tensorflow   |     MLPMixer-huge-14     |   432,350,952   |      =      |
       ---------------------------------------------------------------------------

  # Reference:
    - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf)
    - Source: https://github.com/google-research/vision_transformer
              https://github.com/isaaccorley/mlp-mixer-pytorch

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import (ExtractPatches, MLPBlock, SAMModel, 
                           get_normalizer_from_name, get_activation_from_name)
from utils.model_processing import _obtain_input_shape


class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, activation='gelu', normalizer='layer-norm', norm_eps=1e-6, drop_rate=0.1):
        super(MixerBlock, self).__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.activation = activation
        self.normalizer = normalizer
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.token_mlp_block = MLPBlock(self.tokens_mlp_dim, 
                                        activation=self.activation,
                                        drop_rate=self.drop_rate)
        self.channel_mlp_block = MLPBlock(self.channels_mlp_dim, 
                                          activation=self.activation,
                                          drop_rate=self.drop_rate)
        self.layerNorm1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.layerNorm2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)

    def __token_mixing(self, x):
        # Token-mixing block
        y = tf.transpose(self.layerNorm1(x), perm=(0, 2, 1))  
        y = self.token_mlp_block(y)
        y = tf.transpose(y, perm=(0, 2, 1)) + x
        return y
    
    def __channel_mixing(self, x):
        # Channel-minxing block
        y = self.layerNorm2(x)
        y = self.channel_mlp_block(y) + x
        return y

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        tok = self.__token_mixing(inputs)
        output = self.__channel_mixing(tok)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
                "tokens_mlp_dim": self.tokens_mlp_dim,
                "channels_mlp_dim": self.channels_mlp_dim,
                "norm_eps": self.norm_eps,
                "drop_rate": self.drop_rate
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def MLPMixer(patch_size, 
             num_blocks, 
             hidden_dim, 
             tokens_mlp_dim, 
             channels_mlp_dim, 
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000,
             sam_rho=0.0,
             norm_eps=1e-6,
             drop_rate=0.1):
        
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

    x = ExtractPatches(patch_size, hidden_dim)(img_input)

    for i in range(num_blocks):
        x = MixerBlock(tokens_mlp_dim, channels_mlp_dim, 'gelu', 'layer-norm', norm_eps, drop_rate)(x)
        
    x = get_normalizer_from_name('layer-norm', epsilon=norm_eps)(x)
    
    if include_top:
        x = GlobalAveragePooling1D()(x)
        x = Dropout(rate=drop_rate)(x)
        x = Dense(
            units=1 if num_classes == 2 else num_classes,
            activation=final_activation,
            name="predictions"
        )(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, x, name=name + '_SAM')
        else:
            return Model(inputs, x, name=name)
            
    # Create model.
    if num_blocks == 8:
        model = __build_model(inputs, x, sam_rho, name=f'MLPMixer-S{patch_size}')
    elif num_blocks == 12:
        model = __build_model(inputs, x, sam_rho, name=f'MLPMixer-B{patch_size}')
    elif num_blocks == 24:
        model = __build_model(inputs, x, sam_rho, name=f'MLPMixer-L{patch_size}')
    elif num_blocks == 32:
        model = __build_model(inputs, x, sam_rho, name=f'MLPMixer-H{patch_size}')
    else:
        model = __build_model(inputs, x, sam_rho, name=f'MLPMixer-{patch_size}')

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


def MLPMixer_S16(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=16, 
                     num_blocks=8, 
                     hidden_dim=512, 
                     tokens_mlp_dim=256, 
                     channels_mlp_dim=2048,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_S32(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=32, 
                     num_blocks=8, 
                     hidden_dim=512, 
                     tokens_mlp_dim=256, 
                     channels_mlp_dim=2048,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_B16(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=16, 
                     num_blocks=12, 
                     hidden_dim=768, 
                     tokens_mlp_dim=384, 
                     channels_mlp_dim=3072,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_B32(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=32, 
                     num_blocks=12, 
                     hidden_dim=768, 
                     tokens_mlp_dim=384, 
                     channels_mlp_dim=3072,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_L16(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=16, 
                     num_blocks=24, 
                     hidden_dim=1024, 
                     tokens_mlp_dim=512, 
                     channels_mlp_dim=4096,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_L32(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=32,
                     num_blocks=24, 
                     hidden_dim=1024, 
                     tokens_mlp_dim=512, 
                     channels_mlp_dim=4096,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model


def MLPMixer_H14(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 final_activation="softmax",
                 classes=1000,
                 sam_rho=0.0,
                 norm_eps=1e-6,
                 drop_rate=0.1) -> Model:
    
    model = MLPMixer(patch_size=14,
                     num_blocks=32, 
                     hidden_dim=1280, 
                     tokens_mlp_dim=640, 
                     channels_mlp_dim=5120,
                     include_top=include_top,
                     weights=weights, 
                     input_tensor=input_tensor, 
                     input_shape=input_shape, 
                     pooling=pooling, 
                     final_activation=final_activation,
                     classes=classes,
                     sam_rho=sam_rho,
                     norm_eps=norm_eps,
                     drop_rate=drop_rate)
    return model