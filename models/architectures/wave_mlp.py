"""
  # Description:
    - The following table comparing the params of the WaveMLP in Tensorflow on 
    size 224 x 224 x 3:
      
       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |      WaveMLP-T      |    17,217,736   |
      |---------------------|-----------------|
      |      WaveMLP-S      |    30,729,032   |
      |---------------------|-----------------|
      |      WaveMLP-M      |    44,088,632   |
      |---------------------|-----------------|
      |      WaveMLP-B      |    63,622,456   |
       ---------------------------------------
       
  # Reference:
    - [An Image Patch is a Wave: Phase-Aware Vision MLP](https://arxiv.org/pdf/2111.12294.pdf)
    - Source: https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/mlp_family/wave_mlp.py

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
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import add, multiply, concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import DropPath, MLPBlock, SAMModel, get_activation_from_name, get_nomalizer_from_name
from utils.model_processing import _obtain_input_shape


def phase_aware_token_mixing(inputs, out_dim=-1, qkv_bias=False, output_dropout=0, activation="gelu", name=None):
    out_dim = out_dim if out_dim > 0 else inputs.shape[-1]
    theta_h = Conv2D(filters=out_dim,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     padding="valid",
                     use_bias=True,
                     name=name and name + "theta_h_")(inputs)
    theta_h = get_nomalizer_from_name('batch-norm')(theta_h)
    theta_h = get_activation_from_name('relu')(theta_h)
    height = Conv2D(filters=out_dim,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    use_bias=qkv_bias,
                    name=name and name + "height_")(inputs)

    height_cos = multiply([height, tf.cos(theta_h)])
    height_sin = multiply([height, tf.sin(theta_h)])
    height = concatenate([height_cos, height_sin], axis=-1)
    height = Conv2D(filters=out_dim,
                    kernel_size=(1, 7),
                    strides=(1, 1),
                    padding="same",
                    groups=out_dim,
                    use_bias=False,
                    name=name and name + "height_down_")(height)

    theta_w = Conv2D(filters=out_dim,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     use_bias=True,
                     name=name and name + "theta_w_")(inputs)
    theta_w = get_nomalizer_from_name('batch-norm')(theta_w)
    theta_w = get_activation_from_name('relu')(theta_w)

    width = Conv2D(filters=out_dim,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   use_bias=qkv_bias,
                   name=name and name + "width_")(inputs)

    width_cos = multiply([width, tf.cos(theta_w)])
    width_sin = multiply([width, tf.sin(theta_w)])

    width = concatenate([width_cos, width_sin], axis=-1)
    width = Conv2D(filters=out_dim,
                   kernel_size=(7, 1),
                   strides=(1, 1),
                   padding="same",
                   groups=out_dim,
                   use_bias=False,
                   name=name and name + "width_down_")(width)

    channel = Conv2D(filters=out_dim,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     use_bias=qkv_bias,
                     name=name and name + "channel_")(inputs)

    nn = add([height, width, channel])
    nn = GlobalAveragePooling2D(keepdims=True)(nn)
    nn = MLPBlock(out_dim // 4, 
                  out_dim=out_dim * 3, 
                  use_conv=True, 
                  activation=activation, 
                  name=name and name + "reweight_")(nn)
    nn = Reshape([1, 1, out_dim, 3])(nn)
    nn = Softmax(axis=-1, name=name and name + "attention_scores")(nn)
    attn_height, attn_width, attn_channel = tf.unstack(nn, axis=-1)
    attn_height = multiply([height, attn_height])
    attn_width = multiply([width, attn_width])
    attn_channel = multiply([channel, attn_channel])
    attn = add([attn_height, attn_width, attn_channel])

    out = Conv2D(filters=out_dim,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   use_bias=True,
                   name=name and name + "out_")(attn)
    
    if output_dropout > 0:
        out = Dropout(output_dropout, name=name and name + "out_drop")(out)
    return out


def wave_block(inputs, qkv_bias=False, mlp_ratio=4, drop_prob=0, activation="gelu", normalizer='batch-norm', name=""):
    attn = get_nomalizer_from_name(normalizer)(inputs)
    attn = phase_aware_token_mixing(attn, out_dim=inputs.shape[-1], qkv_bias=qkv_bias, activation=activation, name=name + "attn_")
    attn = DropPath(drop_prob=drop_prob, name=name + "attn_drop_")(attn)
    attn_out = add([inputs, attn], name=name + "attn_out")

    mlp = get_nomalizer_from_name(normalizer)(attn_out)
    mlp = MLPBlock(int(inputs.shape[-1] * mlp_ratio), use_conv=True, activation=activation, name=name + "mlp_blocl_")(mlp)
    mlp = DropPath(drop_prob=drop_prob, name=name + "mlp_drop_")(mlp)
    mlp_out = add([attn_out, mlp], name=name + "mlp_out")
    return mlp_out

    
def WaveMLP(filters,
            num_blocks,
            stem_width,
            mlp_ratios,
            use_downsample_norm,
            norm_name,
            qkv_bias,
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

    stem_width = stem_width if stem_width > 0 else filters[0]
    x = ZeroPadding2D(padding=2, name="stem_pad")(img_input)
    x = Conv2D(filters=stem_width, kernel_size=(7, 7), strides=(4, 4), padding="valid", use_bias=True, name="stem_conv_")(x)

    if use_downsample_norm:
        x = get_nomalizer_from_name(norm_name)(x)

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, filter, mlp_ratio) in enumerate(zip(num_blocks, filters, mlp_ratios)):
        stage_name = "stack{}_".format(stack_id + 1)
        # if stack_id ==1:
        #     break
        if stack_id > 0:
            x = Conv2D(filters=filter, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=True, name=stage_name + "down_sample_")(x)
            
            if use_downsample_norm:
                x = get_nomalizer_from_name(norm_name)(x)

        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            drop_prob = drop_rate * global_block_id / total_blocks
            global_block_id += 1
            x = wave_block(x, qkv_bias, mlp_ratio, drop_prob, activation='gelu', normalizer=norm_name, name=name)
    # return Model(img_input, x, name='abc')
    x = get_nomalizer_from_name(norm_name)(x)
    
    if include_top:
        x = GlobalAveragePooling2D()(x)
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
    if num_blocks == [2, 2, 4, 2] and filters == [64, 128, 320, 512] and mlp_ratios == [4, 4, 4, 4]:
        model = __build_model(inputs, x, sam_rho, name='WaveMLP-T')
    elif num_blocks == [2, 3, 10, 3] and filters == [64, 128, 320, 512] and mlp_ratios == [4, 4, 4, 4]:
        model = __build_model(inputs, x, sam_rho, name='WaveMLP-S')
    elif num_blocks == [3, 4, 18, 3] and filters == [64, 128, 320, 512] and mlp_ratios == [8, 8, 4, 4]:
        model = __build_model(inputs, x, sam_rho, name='WaveMLP-M')
    elif num_blocks == [2, 2, 18, 2] and filters == [96, 192, 384, 768] and mlp_ratios == [4, 4, 4, 4]:
        model = __build_model(inputs, x, sam_rho, name='WaveMLP-B')
    else:
        model = __build_model(inputs, x, sam_rho, name='WaveMLP')

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


def WaveMLP_T(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              drop_rate=0.1) -> Model:
    
    model = WaveMLP(filters=[64, 128, 320, 512],
                    num_blocks=[2, 2, 4, 2],
                    stem_width=-1,
                    mlp_ratios=[4, 4, 4, 4],
                    use_downsample_norm=True,
                    norm_name='batch-norm',
                    qkv_bias=False,
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


def WaveMLP_S(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              drop_rate=0.1) -> Model:
    
    model = WaveMLP(filters=[64, 128, 320, 512],
                    num_blocks=[2, 3, 10, 3],
                    stem_width=-1,
                    mlp_ratios=[4, 4, 4, 4],
                    use_downsample_norm=True,
                    norm_name='group-norm',
                    qkv_bias=False,
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


def WaveMLP_M(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              drop_rate=0.1) -> Model:
    
    model = WaveMLP(filters=[64, 128, 320, 512],
                    num_blocks=[3, 4, 18, 3],
                    stem_width=-1,
                    mlp_ratios=[8, 8, 4, 4],
                    use_downsample_norm=False,
                    norm_name='group-norm',
                    qkv_bias=False,
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


def WaveMLP_B(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              final_activation="softmax",
              classes=1000,
              sam_rho=0.0,
              drop_rate=0.1) -> Model:
    
    model = WaveMLP(filters=[96, 192, 384, 768],
                    num_blocks=[2, 2, 18, 2],
                    stem_width=-1,
                    mlp_ratios=[4, 4, 4, 4],
                    use_downsample_norm=False,
                    norm_name='group-norm',
                    qkv_bias=False,
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