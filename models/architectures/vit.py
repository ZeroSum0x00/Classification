"""
  # Description:
    - The following table comparing the params of the Vision Transformer (ViT) in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     ViT-Base-16     |   86,567,656    |
      |---------------------|-----------------|
      |     ViT-Base-32     |   88,224,232    |
      |---------------------|-----------------|
      |     ViT-Large-16    |   304,326,632   |
      |---------------------|-----------------|
      |     ViT-Large-32    |   306,535,400   |
      |---------------------|-----------------|
      |     ViT-Huge-16     |   632,199,400   |
      |---------------------|-----------------|
      |     ViT-Huge-32     |   634,960,360   |
       ---------------------------------------

  # Reference:
    - [An image is worth 16x16 words: transformers for image recognition 
       at scale](https://arxiv.org/pdf/2010.11929.pdf)
    - Source: https://github.com/faustomorales/vit-keras

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import (ExtractPatches, ClassificationToken, 
                           AddPositionEmbedding, TransformerBlock, SAMModel,
                           get_activation_from_name, get_nomalizer_from_name)
from utils.model_processing import _obtain_input_shape


def ViT(num_layers=12,
        patch_size=16,
        num_heads=12,
        mlp_dim=3072,
        hidden_dim=768,
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ExtractPatches(patch_size, hidden_dim)(img_input)
    x = ClassificationToken(name="class_token")(x)
    x = AddPositionEmbedding(name="Transformer/posembed_input")(x)
    for n in range(num_layers):
        x, _ = TransformerBlock(num_heads=num_heads,
                                mlp_dim=mlp_dim,
                                norm_eps=norm_eps,
                                drop_rate=drop_rate,
                                name=f"Transformer/encoderblock_{n}")(x)
    x = get_nomalizer_from_name('layer-norm', epsilon=norm_eps, name="Transformer/encoder_norm")(x)
    x = Lambda(lambda v: v[:, 0], name="ExtractToken")(x)

    if include_top:
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

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, x, name=name + '_SAM')
        else:
            return Model(inputs, x, name=name)
            
    # Create model.
    if num_layers == 12:
        if patch_size == 16:
            model = __build_model(inputs, x, sam_rho, name='ViT-Base-16')
        elif patch_size == 32:
            model = __build_model(inputs, x, sam_rho, name='ViT-Base-32')
        else:
            model = __build_model(inputs, x, sam_rho, name='ViT-Base')
    elif num_layers == 24:
        if patch_size == 16:
            model = __build_model(inputs, x, sam_rho, name='ViT-Large-16')
        elif patch_size == 32:
            model = __build_model(inputs, x, sam_rho, name='ViT-Large-32')
        else:
            model = __build_model(inputs, x, sam_rho, name='ViT-Large')
    elif num_layers == 32:
        if patch_size == 16:
            model = __build_model(inputs, x, sam_rho, name='ViT-Huge-16')
        elif patch_size == 32:
            model = __build_model(inputs, x, sam_rho, name='ViT-Huge-32')
        else:
            model = __build_model(inputs, x, sam_rho, name='ViT-Huge')
    else:
        model = __build_model(inputs, x, sam_rho, name='ViT')

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


def ViTBase16(include_top=True, 
              weights='imagenet',
              input_tensor=None, 
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              norm_eps=1e-6,
              drop_rate=0.1):

    model = ViT(num_layers=12,
                patch_size=16,
                num_heads=12,
                mlp_dim=3072,
                hidden_dim=768,
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


def ViTBase32(include_top=True, 
              weights='imagenet',
              input_tensor=None, 
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              norm_eps=1e-6,
              drop_rate=0.1):

    model = ViT(num_layers=12,
                patch_size=32,
                num_heads=12,
                mlp_dim=3072,
                hidden_dim=768,
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


def ViTLarge16(include_top=True, 
               weights='imagenet',
               input_tensor=None, 
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               norm_eps=1e-6,
               drop_rate=0.1):

    model = ViT(num_layers=24,
                patch_size=16,
                num_heads=16,
                mlp_dim=4096,
                hidden_dim=1024,
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


def ViTLarge32(include_top=True, 
               weights='imagenet',
               input_tensor=None, 
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               norm_eps=1e-6,
               drop_rate=0.1):

    model = ViT(num_layers=24,
                patch_size=32,
                num_heads=16,
                mlp_dim=4096,
                hidden_dim=1024,
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


def ViTHuge16(include_top=True, 
              weights='imagenet',
              input_tensor=None, 
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              norm_eps=1e-6,
              drop_rate=0.1):

    model = ViT(num_layers=32,
                patch_size=16,
                num_heads=16,
                mlp_dim=5120,
                hidden_dim=1280,
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


def ViTHuge32(include_top=True, 
              weights='imagenet',
              input_tensor=None, 
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              norm_eps=1e-6,
              drop_rate=0.1):

    model = ViT(num_layers=32,
                patch_size=32,
                num_heads=16,
                mlp_dim=5120,
                hidden_dim=1280,
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
