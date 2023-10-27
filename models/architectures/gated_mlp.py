"""
  # Description:
    - The following table comparing the params of the gMLP in Tensorflow on 
    size 224 x 224 x 3:
      
       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |      gMLP-T16       |     5,867,328   |
      |---------------------|-----------------|
      |      gMLP-S16       |    19,422,656   |
      |---------------------|-----------------|
      |      gMLP-B16       |    73,075,392   |
       ---------------------------------------
       
  # Reference:
    - [Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf)
    - Source: https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/gated_mlp.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import add, multiply
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import SAMModel, get_activation_from_name, get_normalizer_from_name
from utils.model_processing import _obtain_input_shape


def spatial_gating_block(inputs, normalizer='layer-norm', name=None):
    xx, yy = tf.split(inputs, 2, axis=-1)
    yy = get_normalizer_from_name(normalizer, name=name and name + "yy_ln")(yy)
    yy = Permute((2, 1), name=name and name + "permute_1")(yy)
    ww_init = tf.initializers.truncated_normal(stddev=1e-6)
    yy = Dense(yy.shape[-1], kernel_initializer=ww_init, bias_initializer="ones", name=name and name + "yy_dense")(yy)
    yy = Permute((2, 1), name=name and name + "permute_2")(yy)
    gated_out = multiply([xx, yy])
    return gated_out


def res_gated_mlp_block(inputs, channels_mlp_dim, drop_rate=0, activation="gelu", normalizer='layer-norm', name=None):
    x = get_normalizer_from_name(normalizer, name=name + "pre_ln")(inputs)
    x = Dense(channels_mlp_dim, name=name + "pre_dense")(x)
    x = get_activation_from_name(activation)(x)
    x = spatial_gating_block(x, normalizer=normalizer, name=name)
    x = Dense(inputs.shape[-1], name=name + "gated_dense")(x)

    if drop_rate > 0:
        x = Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(x)
    return add([x, inputs])

    
def gMLP(stem_width,
         patch_size,
         num_blocks,
         channels_mlp_dim,
         include_top=True,
         weights='imagenet',
         input_tensor=None,
         input_shape=None,
         pooling=None,
         final_activation="softmax",
         classes=1000,
         sam_rho=0.0,
         drop_rate=0):
        
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

    x = Conv2D(filters=stem_width, 
               kernel_size=patch_size, 
               strides=patch_size, 
               padding="valid", 
               name="stem")(img_input)
    x = Reshape(target_shape=(-1, stem_width))(x)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_rate, (list, tuple)) else [drop_rate, drop_rate]
    
    for ii in range(num_blocks):
        block_name = "{}_{}_".format("gmlp", str(ii + 1))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        x = res_gated_mlp_block(x, channels_mlp_dim=channels_mlp_dim, drop_rate=block_drop_rate, activation='gelu', name=block_name)
        
    x = get_normalizer_from_name('layer-norm', name="pre_head_norm")(x)
             
    if include_top:
        x = GlobalAveragePooling1D()(x)
        x = Dropout(rate=drop_rate)(x)
        x = Dense(1 if classes == 2 else classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
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
    if stem_width == 128:
        model = __build_model(inputs, x, sam_rho, name=f'gMLP-T{patch_size}')
    elif stem_width == 256:
        model = __build_model(inputs, x, sam_rho, name=f'gMLP-S{patch_size}')
    elif stem_width == 512:
        model = __build_model(inputs, x, sam_rho, name=f'gMLP-B{patch_size}')
    else:
        model = __build_model(inputs, x, sam_rho, name=f'gMLP-{patch_size}')

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


def gMLP_T16(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000,
             sam_rho=0.0,
             drop_rate=0.1) -> Model:
    
    model = gMLP(stem_width=128,
                 patch_size=16,
                 num_blocks=30,
                 channels_mlp_dim=128*6,
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


def gMLP_S16(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000,
             sam_rho=0.0,
             drop_rate=0.1) -> Model:
    
    model = gMLP(stem_width=256,
                 patch_size=16,
                 num_blocks=30,
                 channels_mlp_dim=256*6,
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


def gMLP_B16(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             final_activation="softmax",
             classes=1000,
             sam_rho=0.0,
             drop_rate=0.1) -> Model:
    
    model = gMLP(stem_width=512,
                 patch_size=16,
                 num_blocks=30,
                 channels_mlp_dim=512*6,
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