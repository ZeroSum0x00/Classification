"""
  # Description:
    - The following table comparing the params of the Swin Transformer (Swin) in Tensorflow on 
    size 224 x 224 x 3:

       -------------------------------------------------
      |           Model Name          |    Params       |
      |-------------------------------------------------|
      |     Swin Transformer tiny     |   28,538,058    |
      |-------------------------------------------------|
      |     Swin Transformer small    |   49,942,398    |
      |-------------------------------------------------|
      |     Swin Transformer base     |   88,104,364    |
      |-------------------------------------------------|
      |     Swin Transformer large    |  196,868,616    |
       -------------------------------------------------

  # Reference:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
    - Source: https://github.com/microsoft/Swin-Transformer

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.utils import get_source_inputs, get_file
from models.layers import MLPBlock, DropPath
from utils.model_processing import _obtain_input_shape


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, shape=[-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(window, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        C (int): Channel of image

    Returns:
        x: (B, H, W, C)
    """
    x = tf.reshape(window, shape=[-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., *args, **kwargs):
        super(WindowAttention, self).__init__(*args, **kwargs)
        self.dim         = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_heads   = num_heads
        self.qkv_bias    = qkv_bias
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5
        self.attn_drop   = attn_drop
        self.proj_drop   = proj_drop

    def build(self, input_shape):
        self.qkv_projection     = Dense(self.dim * 3)
        self.attention_dropout  = Dropout(self.attn_drop)
        self.projection         = Dense(self.dim)
        self.projection_dropout = Dropout(self.proj_drop)

        self.relative_position_bias_table = self.add_weight(
            f'attn/relative_position_bias_table',
            shape       = ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer = tf.initializers.zeros(),
            trainable   = True
        )
        coords_h        = np.arange(self.window_size[0])
        coords_w        = np.arange(self.window_size[1])
        coords          = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten  = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        relative_position_tensor = tf.convert_to_tensor(relative_position_index)
        self.relative_position_index = tf.Variable(
            initial_value=relative_position_tensor, trainable=False, name=f'attn/relative_position_index'
        )

    def call(self, inputs, mask=None):
        B_, N, C = inputs.shape
        qkv      = self.qkv_projection(inputs)

        qkv      = tf.reshape(qkv, shape=[-1, N, 3, self.num_heads, C // self.num_heads])
        qkv      = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v  = qkv[0], qkv[1], qkv[2]
        q        = q * self.scale
        attn     = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))

        relative_position_bias = tf.gather(self.relative_position_bias_table,
                                           tf.reshape(self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=[self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + \
                tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)

            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attention_dropout(attn)
        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        proj_drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path_prob (float, optional): Stochastic depth rate. Default: 0.0
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, activation='gelu', proj_drop=0., attn_drop=0., drop_path_prob=0., *args, **kwargs):
        super(SwinTransformerBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.activation = activation
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        self.norm_layer1 = LayerNormalization(epsilon=1e-5)
        self.attention   = WindowAttention(dim=self.dim,
                                           window_size=self.window_size,
                                           num_heads=self.num_heads,
                                           qkv_bias=self.qkv_bias,
                                           qk_scale=self.qk_scale,
                                           attn_drop=self.attn_drop,
                                           proj_drop=self.proj_drop)

        self.drop_path   = DropPath(self.drop_path_prob if self.drop_path_prob > 0. else 0.)
        self.norm_layer2 = LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim   = int(self.dim * self.mlp_ratio)
        self.mlp         = MLPBlock(mlp_dim=mlp_hidden_dim,
                                    activation=self.activation,
                                    drop_rate=self.proj_drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0,                 -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size,   None))
            w_slices = (slice(0,                 -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size,   None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows,
                                      shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(tf.not_equal(attn_mask, 0), -100.0 * tf.ones_like(attn_mask), attn_mask)
            attn_mask = tf.where(tf.equal(attn_mask, 0), tf.zeros_like(attn_mask), attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)
        else:
            self.attn_mask = None

    def call(self, inputs):
        B, L, C = inputs.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"

        shortcut = inputs
        x = self.norm_layer1(inputs)
        x = tf.reshape(x, shape=[-1, H, W, C])

        pad_left = pad_top = 0
        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size
        pad = tf.constant([[0,        0,],
                           [pad_top,  pad_bottom],
                           [pad_left, pad_right],
                           [0,        0]])
        x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)
        _, Hp, Wp, _ = x.shape
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :H, :W, :]

        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm_layer2(x)))
        return x


def BasicLayer(
    x, dim, input_resolution, depth, num_heads, window_size,
    mlp_ratio=4., qkv_bias=True, qk_scale=None, activation='gelu', drop=0., attn_drop=0., drop_path_prob=0., name=""
):
    for i in range(depth):
        x = SwinTransformerBlock(
                    dim                 = dim,
                    input_resolution    = input_resolution,
                    num_heads           = num_heads,
                    window_size         = window_size,
                    shift_size          = 0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio           = mlp_ratio,
                    qkv_bias            = qkv_bias,
                    qk_scale            = qk_scale,
                    activation          = activation,
                    proj_drop           = drop,
                    attn_drop           = attn_drop,
                    drop_path_prob      = drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob)(x)
    return x


class PatchMerging(tf.keras.layers.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
    """
    
    def __init__(self, input_resolution, *args, **kwargs):
        super(PatchMerging, self).__init__(*args, **kwargs)
        self.input_resolution = input_resolution

    def build(self, input_shape):
        self.norm_layer = LayerNormalization(epsilon=1e-5)
        self.projection = Dense(2 * input_shape[-1], use_bias=False)

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        if (H % 2 == 1) or (W % 2 == 1):
            pad = tf.constant([[0,     0,],
                               [0,     0],
                               [W % 2, H % 2],
                               [0,     0]])
            x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])
        x = self.norm_layer(x)
        x = self.projection(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):
    r""" Image to Patch Embedding

    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
        patch_size (int): Patch token size. Default: 4.
        drop_rate (float): Dropout rate. Default: 0
    """
    def __init__(self, embed_dim, patch_size, drop_rate=0., *args, **kwargs):
        super(PatchEmbed, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.projection = Conv2D(self.embed_dim,
                                 kernel_size=self.patch_size,
                                 strides=self.patch_size,
                                 padding="valid")
        self.norm_layer = LayerNormalization(epsilon=1e-5)
        self.reshape_tensor = Reshape(((input_shape[1] // self.patch_size[0]) * (input_shape[2] // self.patch_size[0]), self.embed_dim))
        self.drop_out = Dropout(self.drop_rate)
        if input_shape[2] % self.patch_size[1] != 0:
            self.width_pad = self.patch_size[1] - W % self.patch_size[1]
        else:
            self.width_pad = None

        if input_shape[2] % self.patch_size[1] != 0:
            self.height_pad = self.patch_size[0] - H % self.patch_size[0]
        else:
            self.height_pad = None
            
    def call(self, inputs, training=False):
        if self.width_pad:
            pad = tf.constant([[0,              0],
                               [0,              0],
                               [self.width_pad, 0],
                               [0,              0]])
            inputs = tf.pad(inputs, pad, mode='CONSTANT', constant_values=0)

        if self.height_pad:
            pad = tf.constant([[0,               0],
                               [0,               0],
                               [self.height_pad, 0],
                               [0,               0]])
            inputs = tf.pad(inputs, pad, mode='CONSTANT', constant_values=0)

        x = self.projection(inputs, training=training)
        x = self.reshape_tensor(x)
        x = self.norm_layer(x)
        x = self.drop_out(x)
        return x


def Swin(embed_dim=96,
         patch_size=(4, 4),
         num_heads=[3, 6, 12, 24],
         depths=[2, 2, 6, 2],
         window_size=7,
         mlp_ratio=4.0,
         qkv_bias=True,
         qk_scale=None,
         include_top=True,
         weights='imagenet',
         input_tensor=None,
         input_shape=None,
         pooling=None,
         final_activation="softmax",
         classes=1000,
         drop_rate=0.0,
         attn_drop_rate=0.0,
         drop_path_rate=0.1):
             
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

    x = PatchEmbed(embed_dim, patch_size, drop_rate=drop_rate)(img_input)
    patches_resolution  = [input_shape[0] // patch_size[0], input_shape[1] // patch_size[1]]
    dpr                 = [x for x in np.linspace(0., drop_path_rate, sum(depths))]
    num_layers          = len(depths)
    for i, (head, depth) in enumerate(zip(num_heads, depths)):
        dim = int(embed_dim * 2 ** i)
        input_resolution    = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
        x = BasicLayer(x,
                       dim              = dim,
                       input_resolution = input_resolution,
                       depth            = depth,
                       num_heads        = head,
                       window_size      = window_size,
                       mlp_ratio        = mlp_ratio,
                       qkv_bias         = qkv_bias,
                       qk_scale         = qk_scale,
                       activation       = 'gelu',
                       drop             = drop_rate,
                       attn_drop        = attn_drop_rate,
                       drop_path_prob   = dpr[sum(depths[:i]):sum(depths[:i + 1])],
        )
        if (i < num_layers - 1):
            x   = PatchMerging(input_resolution)(x)
            
    x = LayerNormalization(epsilon=1e-5)(x)

    if include_top:
        x = GlobalAveragePooling1D()(x)
        x = Dense(1 if classes == 2 else classes, activation=final_activation, name='predictions')(x)
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

    # Create model.
    if num_heads == [3, 6, 12, 24] and depths == [2, 2, 6, 2]:
        model = Model(inputs, x, name='SwinTransformer-Tiny')
    elif num_heads == [3, 6, 12, 24] and depths == [2, 2, 18, 2]:
        model = Model(inputs, x, name='SwinTransformer-Small')
    elif num_heads == [4, 8, 16, 32] and depths == [2, 2, 18, 2]:
        model = Model(inputs, x, name='SwinTransformer-Base')
    elif num_heads == [6, 12, 24, 48] and depths == [2, 2, 18, 2]:
        model = Model(inputs, x, name='SwinTransformer-Large')
    else:
        model = Model(inputs, x, name='SwinTransformer')

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


def SwinT(include_top=True, 
          weights='imagenet',
          input_tensor=None, 
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000,
          drop_rate=0.0,
          attn_drop_rate=0.0,
          drop_path_rate=0.2):

    model = Swin(embed_dim=96,
                 patch_size=(4, 4),
                 num_heads=[3, 6, 12, 24],
                 depths=[2, 2, 6, 2],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 include_top=include_top,
                 weights=weights, 
                 input_tensor=input_tensor, 
                 input_shape=input_shape, 
                 pooling=pooling, 
                 final_activation=final_activation,
                 classes=classes,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_rate)
    return model


def SwinS(include_top=True, 
          weights='imagenet',
          input_tensor=None, 
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000,
          drop_rate=0.0,
          attn_drop_rate=0.0,
          drop_path_rate=0.3):

    model = Swin(embed_dim=96,
                 patch_size=(4, 4),
                 num_heads=[3, 6, 12, 24],
                 depths=[2, 2, 18, 2],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 include_top=include_top,
                 weights=weights, 
                 input_tensor=input_tensor, 
                 input_shape=input_shape, 
                 pooling=pooling, 
                 final_activation=final_activation,
                 classes=classes,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_rate)
    return model


def SwinB(include_top=True, 
          weights='imagenet',
          input_tensor=None, 
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000,
          drop_rate=0.0,
          attn_drop_rate=0.0,
          drop_path_rate=0.5):

    model = Swin(embed_dim=128,
                 patch_size=(4, 4),
                 num_heads=[4, 8, 16, 32],
                 depths=[2, 2, 18, 2],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 include_top=include_top,
                 weights=weights, 
                 input_tensor=input_tensor, 
                 input_shape=input_shape, 
                 pooling=pooling, 
                 final_activation=final_activation,
                 classes=classes,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_rate)
    return model


def SwinL(include_top=True, 
          weights='imagenet',
          input_tensor=None, 
          input_shape=None,
          pooling=None,
          final_activation="softmax",
          classes=1000,
          drop_rate=0.0,
          attn_drop_rate=0.0,
          drop_path_rate=0.5):

    model = Swin(embed_dim=192,
                 patch_size=(4, 4),
                 num_heads=[6, 12, 24, 48],
                 depths=[2, 2, 18, 2],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 include_top=include_top,
                 weights=weights, 
                 input_tensor=input_tensor, 
                 input_shape=input_shape, 
                 pooling=pooling, 
                 final_activation=final_activation,
                 classes=classes,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_rate)
    return model