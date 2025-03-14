"""
  # Description:
    - The following table comparing the params of the Vision Transformer (ViT) in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     ViT-Base-16     |   86,604,520    |
      |---------------------|-----------------|
      |     ViT-Base-32     |   88,261,096    |
      |---------------------|-----------------|
      |     ViT-Large-16    |   304,424,936   |
      |---------------------|-----------------|
      |     ViT-Large-32    |   306,633,704   |
      |---------------------|-----------------|
      |     ViT-Huge-16     |   632,363,240   |
      |---------------------|-----------------|
      |     ViT-Huge-32     |   635,124,200   |
       ---------------------------------------

  # Reference:
    - [An image is worth 16x16 words: transformers for image recognition 
       at scale](https://arxiv.org/pdf/2010.11929.pdf)
    - Source: https://github.com/faustomorales/vit-keras

"""

from __future__ import print_function
from __future__ import absolute_import

import copy
import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import (ExtractPatches, ClassificationToken, 
                           MultiHeadSelfAttention, MLPBlock,
                           PositionalEmbedding, TransformerBlock, SAMModel,
                           get_activation_from_name, get_normalizer_from_name)
from utils.model_processing import _obtain_input_shape


def ViT(attention_block=None,
        mlp_block=None,
        num_layers=12,
        patch_size=16,
        num_heads=12,
        mlp_dim=3072,
        hidden_dim=768,
        include_top=True, 
        weights='imagenet',
        input_tensor=None, 
        input_shape=None,
        pooling=None,
        activation='gelu',
        normalizer='layer-norm',
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
    x = PositionalEmbedding(name="Posembed_input")(x)

    for n in range(num_layers):
        if attention_block is None:
            attn_clone = MultiHeadSelfAttention(num_heads=num_heads,
                                                return_weight=False,
                                                name=f"MultiHeadDotProductAttention_{n}")
        else:
            attn_clone = copy.deepcopy(attention_block)
            
        if mlp_block is None:
            mlp_clone = MLPBlock(mlp_dim,
                                 activation=activation,
                                 normalizer=normalizer, 
                                 drop_rate=drop_rate, 
                                 name=f"MlpBlock_{n}")
        else:
            mlp_clone = copy.deepcopy(mlp_block)

        x, _ = TransformerBlock(attn_clone,
                                mlp_clone,
                                activation=activation,
                                norm_eps=norm_eps,
                                drop_rate=drop_rate,
                                name=f"encoderblock_{n}")(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="encoder_norm")(x)
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
        model = __build_model(inputs, x, sam_rho, name=f'ViT-Base-{patch_size}')
    elif num_layers == 24:
        model = __build_model(inputs, x, sam_rho, name=f'ViT-Large-{patch_size}')
    elif num_layers == 32:
        model = __build_model(inputs, x, sam_rho, name=f'ViT-Huge-{patch_size}')
    else:
        model = __build_model(inputs, x, sam_rho, name=f'ViT-{patch_size}')

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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=12,
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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=12,
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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=24,
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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=24,
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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=32,
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

    model = ViT(attention_block=None,
                mlp_block=None,
                num_layers=32,
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