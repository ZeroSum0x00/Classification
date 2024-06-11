"""
  # Description:
    - The following table comparing the params of the Vision Mamba (Vim) in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------
      |     Model Name     |    Params       |
      |--------------------------------------|
      |      Vim-Base      |   8,543,720     |
       ---------------------------------------
       
  # Reference:
    - [Vision Mamba: Efficient Visual Representation Learning 
       with Bidirectional State Space Model](https://arxiv.org/pdf/2401.09417)
       
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import (ExtractPatches, SSM, SAMModel,
                           get_activation_from_name, get_normalizer_from_name)
from utils.model_processing import _obtain_input_shape


class MambaEncoderBlock(tf.keras.layers.Layer):

    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    args:
      dim (int): Dimension of the model.
      dt_rank (int): The rank of the state space model.
      dim_inner (int): The dimension of the inner layer of the multi-head attention.
      d_state (int): The dimension of the state space model.
      activation (str): activation name.
      norm_layer (str): normalization name.

    returns:
      output: result of the Vision Mamba block

    """

    def __init__(self,
                 dim,
                 dt_rank,
                 dim_inner,
                 d_state,
                 activation='silu',
                 norm_layer='layer-norm',
                 *args,
                 **kwargs):
        super(MambaEncoderBlock, self).__init__(*args, **kwargs)
        self.dim        = dim
        self.dt_rank    = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.activation    = activation
        self.norm_layer    = norm_layer

    def build(self, input_shape):
        self.norm_layer = get_normalizer_from_name(self.norm_layer)
        self.activation = get_activation_from_name(self.activation)
        self.proj = Dense(units=self.dim)
        self.forward_conv1d = Conv1D(filters=self.dim, kernel_size=1, strides=1, padding='valid')
        self.backward_conv1d = Conv1D(filters=self.dim, kernel_size=1, strides=1, padding='valid')
        self.ssm1 = SSM(self.dt_rank, self.dim_inner, self.d_state)
        self.ssm2 = SSM(self.dt_rank, self.dim_inner, self.d_state)

    def process_direction(self, x, bottleneck, ssm):
        x = bottleneck(x)
        x = tf.nn.softplus(x)
        x = ssm(x)
        return x

    def call(self, inputs, training=False):
        skip = inputs
        x = self.norm_layer(inputs, training=training)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x, training=training)
        z = self.activation(z1)

        x = self.proj(x, training=training)
        x1 = self.process_direction(x, self.forward_conv1d, self.ssm1)
        x1 = tf.multiply(x1, z)
        x2 = self.process_direction(x, self.backward_conv1d, self.ssm2)
        x2 = tf.multiply(x2, z)
        return x1 + x2 + skip


def Vim(dim=256,
        dt_rank=32,
        dim_inner=256,
        d_state=256,
        hidden_dim=768,
        patch_size=16,
        num_layers=12,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        activation='silu',
        norm_layer='layer-norm',
        final_activation="softmax",
        classes=1000,
        sam_rho=0.0,
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

    x = ExtractPatches(patch_size, hidden_dim, name="Extract_Patches")(img_input)
    x = Dense(dim)(x)
    x = Dropout(drop_rate)(x)

    for n in range(num_layers):
        x = MambaEncoderBlock(dim=dim,
                              dt_rank=dt_rank,
                              dim_inner=dim_inner,
                              d_state=d_state,
                              activation=activation,
                              norm_layer=norm_layer,
                              name=f'VisionMamba/encoderblock_{n}')(x)

    x = tf.reduce_mean(x, axis=1)
    x = get_normalizer_from_name(norm_layer)(x)

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
    model = __build_model(inputs, x, sam_rho, name='Vision-Mamba')

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


def Vim_Base(include_top=True, 
            weights='imagenet',
            input_tensor=None, 
            input_shape=None,
            pooling=None,
            final_activation="softmax",
            classes=1000,
            sam_rho=0.0,
            drop_rate=0.1):

    model = Vim(dim=256,
                dt_rank=32,
                dim_inner=256,
                d_state=256,
                hidden_dim=768,
                patch_size=16,
                num_layers=12,
                include_top=include_top,
                weights=weights, 
                input_tensor=input_tensor,
                input_shape=input_shape,
                pooling=pooling, 
                final_activation=final_activation,
                classes=classes,
                sam_rho=sam_rho,
                drop_rate=drop_rate)
    return model