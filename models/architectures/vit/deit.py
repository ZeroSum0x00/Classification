"""
  # Description:
    - The following table comparing the params of the Data-efficient image Transformers (DeiT) in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------
      |     Model Name     |    Params       |
      |--------------------------------------|
      |     DeiT-Tiny      |   16,619,792    |
      |--------------------|-----------------|
      |     DeiT-Small     |   36,665,936    |
      |--------------------|-----------------|
      |     DeiT-Base      |   87,375,056    |
       --------------------------------------
       
  # Reference:
    - [Training data-efficient image transformers
       & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)
       
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
                           PositionalEmbedding, TransformerBlock,
                           DistillationToken, SAMModel, 
                           get_normalizer_from_name, get_activation_from_name)
from utils.model_processing import _obtain_input_shape



def DeiT(attention_block=None,
         mlp_block=None,
         num_layers=12,
         patch_size=16,
         num_heads=6,
         mlp_dim=3072,
         hidden_dim=384,
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
         drop_rate=0.1,
         training=False):
         
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

    x = ExtractPatches(patch_size, hidden_dim, name="Extract_Patches")(img_input)
    x = DistillationToken(name="Distillation_Token")(x)
    x = ClassificationToken(name="Classification_Token")(x)
    x = PositionalEmbedding(name="Positional_Embedding")(x)

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
                                name=f"Transformer.encoderblock_{n}")(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="Transformer/encoder_norm")(x)
    x_head = Lambda(lambda v: v[:, 0], name="Extract_Predict_Token")(x)
    x_dist = Lambda(lambda v: v[:, 1], name="Extract_Distillation_Token")(x)

    if include_top:
        x_head = Dense(1 if classes == 2 else classes, name='head')(x_head)
        x_head = get_activation_from_name(final_activation)(x_head)
        
        x_dist = Dense(
            units=1 if num_classes == 2 else num_classes,
            activation=final_activation,
            name="dist"
        )(x_dist)
    else:
        if pooling == 'avg':
            x_head = GlobalAveragePooling2D(name='head_global_avgpool')(x_head)
            x_dist = GlobalAveragePooling2D(name='dist_global_avgpool')(x_dist)
        elif pooling == 'max':
            x_head = GlobalMaxPooling2D(name='head_global_maxpool')(x_head)
            x_dist = GlobalMaxPooling2D(name='dist_global_maxpool')(x_dist)

    if training:
        x = x_head, x_dist
    else:
        x = (x_head + x_dist) / 2

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
    if num_heads == 3 and hidden_dim == 192:
        model = __build_model(inputs, x, sam_rho, name='DeiT-Tiny')
    elif num_heads == 6 and hidden_dim == 384:
        model = __build_model(inputs, x, sam_rho, name='DeiT-Small')
    elif num_heads == 12 and hidden_dim == 768:
        model = __build_model(inputs, x, sam_rho, name='DeiT-Base')
    else:
        model = __build_model(inputs, x, sam_rho, name='DeiT')
             
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


def DeiT_Ti(include_top=True, 
            weights='imagenet',
            input_tensor=None, 
            input_shape=None,
            pooling=None,
            final_activation="softmax",
            classes=1000,
            sam_rho=0.0,
            norm_eps=1e-6,
            drop_rate=0.1):

    model = DeiT(attention_block=None,
                 mlp_block=None,
                 num_layers=12,
                 patch_size=16,
                 num_heads=3,
                 mlp_dim=3072,
                 hidden_dim=192,
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

def DeiT_S(include_top=True, 
           weights='imagenet',
           input_tensor=None, 
           input_shape=None,
           pooling=None,
           final_activation="softmax",
           classes=1000,
           sam_rho=0.0,
           norm_eps=1e-6,
           drop_rate=0.1):

    model = DeiT(attention_block=None,
                 mlp_block=None,
                 num_layers=12,
                 patch_size=16,
                 num_heads=6,
                 mlp_dim=3072,
                 hidden_dim=384,
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

                
def DeiT_B(include_top=True, 
           weights='imagenet',
           input_tensor=None, 
           input_shape=None,
           pooling=None,
           final_activation="softmax",
           classes=1000,
           sam_rho=0.0,
           norm_eps=1e-6,
           drop_rate=0.1):

    model = DeiT(attention_block=None,
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