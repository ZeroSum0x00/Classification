"""
  # Description:
    - The following table comparing the params of the Data-efficient image Transformers (DeiT) in Tensorflow on 
    size 224 x 224 x 3:

  # Reference:
    - [Training data-efficient image transformers
       & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)
       
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
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import ExtractPatches, ClassificationToken, AddPositionEmbedding, TransformerBlock
from utils.model_processing import _obtain_input_shape


def DeiT(patch_size=16,
         num_heads=12,
         num_layers=12,
         mlp_dim=3072,
         hidden_size=768,
         include_top=True, 
         weights='imagenet',
         input_tensor=None, 
         input_shape=None,
         pooling=None,
         final_activation="softmax",
         classes=1000,
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

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ExtractPatches(patch_size, hidden_size, name="Extract_Patches")(img_input)
    x = DistillationToken(name="Distillation_Token")(x)
    x = ClassificationToken(name="Classification_Token")(x)
    x = AddPositionEmbedding(name="Add_Position_Embedding")(x)
    for n in range(num_layers):
        x, _ = TransformerBlock(num_heads=num_heads,
                                mlp_dim=mlp_dim,
                                drop_rate=drop_rate,
                                name=f"Transformer/encoderblock_{n}")(x)
    x = LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x_head = Lambda(lambda v: v[:, 0], name="Extract_Predict_Token")(x)
    x_dist = Lambda(lambda v: v[:, 1], name="Extract_Distillation_Token")(x)

    if include_top:
        x_head = Dense(1 if classes == 2 else classes, activation=final_activation, name='head')(x_head)
        x_dist = Dense(1 if classes == 2 else classes, activation=final_activation, name='dist')(x_dist)
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
        
    # Create model.
    model = Model(inputs=inputs, outputs=x, name='DeiT')

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