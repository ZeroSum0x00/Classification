"""
  # Description:
    - The following table comparing the params of the ViTImageEncoder (Segment Anything Model) in Tensorflow on 
    size 1024 x 1024 x 3:

       -------------------------------------------
      |        Model Name       |    Params       |
      |-------------------------------------------|
      |     ViTImageEncoder     |   86,689,512    |
       -------------------------------------------

  # Reference:
    - [Segment Anything](https://ai.meta.com/research/publications/segment-anything/)
    - Source: https://github.com/facebookresearch/segment-anything

"""

from __future__ import print_function
from __future__ import absolute_import

import math
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.utils import get_source_inputs, get_file

from models.layers import get_activation_from_name, get_normalizer_from_name, PositionalEmbedding, MLPBlock
from utils.model_processing import _obtain_input_shape



def nonoverlap_window_partition(x, window_size):
    
    """
        Partition into non-overlapping windows with padding if needed.
    
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size
            
        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
    """
    
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        pad = tf.constant([[0, 0,],
                           [0, pad_h],
                           [0, pad_w],
                           [0, 0]])
        x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)

    Hp, Wp = H + pad_h, W + pad_w
    x = tf.reshape(x, shape=[-1, Hp // window_size, window_size, Wp // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows, (Hp, Wp)


def nonoverlap_window_reverse(window, window_size, image_size, padding_size):
    """
        Window unpartition into original sequences and removing padding.
        
        Args:
            windows: input tokens with [B * num_windows, window_size, window_size, C].
            window_size (int): window size.
            image_size  (tuple(int, int) or list(int, int)): patches size of before layers [height, width].
            padding_size  (tuple(int, int) or list(int, int)): padded patches size of before layers [height, width].
    
        Returns:
            x: unpartitioned sequences with [B, H, W, C].
    """
    H, W = image_size
    Hp, Wp = padding_size
    C = window.shape[-1]
    
    x = tf.reshape(window, shape=[-1, Hp // window_size, Wp // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, Hp, Wp, C])

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x

    
class PatchEmbed(tf.keras.layers.Layer):

    """
        Image to Patch Embedding.
    
        Args:
            patch_size (int): Size of patch.
            hidden_dim (int): Number of convolutional filters.
            
        Returns:
            patches: Tensor patch after embedding [batch_size, height // patch_size, width // patch_size, hidden_dim].
    """
    
    def __init__(self, patch_size, hidden_dim, *args, **kwargs):
        super(PatchEmbed, self).__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.extractor = Conv2D(filters=self.hidden_dim,
                                kernel_size=self.patch_size,
                                strides=self.patch_size,
                                padding="valid",
                                name="embedding")
        
    def call(self, inputs, training=False):
        x = self.extractor(inputs, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AddingDecomposedRelationPos(tf.keras.layers.Layer):

    """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
        
        Args:
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Inputs:
            inputs (Tensor): attention map.
            query (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            
        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
    """

    def __init__(self, q_size, k_size, *args, **kwargs):
        super(AddingDecomposedRelationPos, self).__init__(*args, **kwargs)
        self.q_size = q_size
        self.k_size = k_size

    def _get_rel_pos(self, rel_pos, q_size, k_size):
        """
            Get relative positional embeddings according to the relative positions of query and key sizes.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)

        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos = tf.expand_dims(rel_pos, axis=0)
            rel_pos = tf.transpose(rel_pos, perm=[1, 2, 0])
            rel_pos_resized = tf.image.resize(rel_pos, (max_rel_dist, rel_pos.shape[1]), tf.image.ResizeMethod.BILINEAR)        
            rel_pos_resized = tf.squeeze(rel_pos_resized, axis=-1)
        else:
            rel_pos_resized = rel_pos
        # Scale the coords with short length if shapes for q and k are different.
        q_coords = np.arange(q_size)
        q_coords = np.expand_dims(q_coords, axis=-1) * max(k_size / q_size, 1.0)

        k_coords = np.arange(k_size)
        k_coords = np.expand_dims(k_coords, axis=0) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        relative_coords = relative_coords.astype(np.int32)
        return tf.gather(rel_pos_resized, relative_coords)

    def call(self, inputs, query, pos_h, pos_w, training=False):
        q_h, q_w = self.q_size
        k_h, k_w = self.k_size
        Rh = self._get_rel_pos(pos_h, q_h, k_h)
        Rw = self._get_rel_pos(pos_w, q_w, k_w)

        B, _, dim = query.shape
        r_q = tf.reshape(query, shape=(-1, q_h, q_w, dim))
        rel_h = tf.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = tf.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = tf.reshape(inputs, shape=(-1, q_h, q_w, k_h, k_w))
        attn = attn + rel_h[..., tf.newaxis] + rel_w[:, :, :, tf.newaxis, :]
        attn = tf.reshape(attn, shape=(-1, q_h * q_w, k_h * k_w))
        return attn


class WindowAttention(tf.keras.layers.Layer):

    """
        Multi-head Attention block with relative position embeddings.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
    """
    
    def __init__(self, dim, num_heads, qkv_bias=True, use_rel_pos=False, *args, **kwargs):
        super(WindowAttention, self).__init__(*args, **kwargs)
        self.dim         = dim
        self.num_heads   = num_heads
        self.qkv_bias    = qkv_bias
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.use_rel_pos = use_rel_pos
        
    def build(self, input_shape):
        q_size = k_size = input_shape[1:3]
        self.qkv_projection     = Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.projection         = Dense(self.dim)
        self.attention          = AddingDecomposedRelationPos(q_size, k_size)
        if self.use_rel_pos:
            self.rel_pos_h = self.add_weight(
                f'attn/relative_position_height',
                shape       = ((2 * input_shape[1] - 1), self.head_dim),
                initializer = tf.initializers.zeros(),
                trainable   = True
            )
            self.rel_pos_w = self.add_weight(
                f'attn/relative_position_width',
                shape       = ((2 * input_shape[2] - 1), self.head_dim),
                initializer = tf.initializers.zeros(),
                trainable   = True
            )
            
    def call(self, inputs, training=False):
        B_, H, W, C = inputs.shape
        qkv         = self.qkv_projection(inputs, training=training)
        qkv         = tf.reshape(qkv, shape=[-1, H * W, 3, self.num_heads, C // self.num_heads])        
        qkv         = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        qkv         = tf.reshape(qkv, shape=[3, -1 , H * W, C // self.num_heads])
        q, k, v     = qkv[0], qkv[1], qkv[2]
        attn        = (q * self.scale) @ tf.transpose(k, perm=[0, 2 ,1])

        if self.use_rel_pos:
            attn = self.attention(attn, q, self.rel_pos_h, self.rel_pos_w)

        attn = tf.nn.softmax(attn, axis=-1)
        x = attn @ v
        x = tf.reshape(x, shape=[-1, self.num_heads, H, W, C // self.num_heads])
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
        x = tf.reshape(x, shape=[-1, H, W, C])
        x = self.projection(x, training=training)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):

    """
        Transformer blocks with support of window attention and residual propagation blocks, according to Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            activation (str or object): Activation layer.
            normalizer (str or object): Normalization layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            proj_drop (float, optional): Dropout rate. Default: 0.0
    """
    
    def __init__(self, dim, num_heads, window_size=0, mlp_ratio=4., qkv_bias=True, activation='gelu', normalizer='layer-norm', use_rel_pos=False, proj_drop=0., *args, **kwargs):
        super(SwinTransformerBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.activation = activation
        self.normalizer = normalizer
        self.use_rel_pos = use_rel_pos
        self.proj_drop  = proj_drop

    def build(self, input_shape):
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=1e-5)
        self.attention   = WindowAttention(dim=self.dim,
                                           num_heads=self.num_heads,
                                           qkv_bias=self.qkv_bias,
                                           use_rel_pos=self.use_rel_pos)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=1e-5)
        mlp_hidden_dim   = int(self.dim * self.mlp_ratio)
        self.mlp         = MLPBlock(mlp_dim=mlp_hidden_dim,
                                    activation=self.activation,
                                    drop_rate=self.proj_drop)
        
    def call(self, inputs, training=False):
        shortcut = inputs
        x = self.norm_layer1(inputs, training=training)
        
        if self.window_size > 0:
            H, W = x.shape[1:3]
            x, pad_hw = nonoverlap_window_partition(x, self.window_size)
            
        x = self.attention(x, training=training)

        if self.window_size > 0:
            x = nonoverlap_window_reverse(x, self.window_size, (H, W), pad_hw)

        x = add([shortcut, x])
        y = self.norm_layer2(x, training=training)
        y = self.mlp(y, training=training)
        return x + y


def BasicLayer(
    x, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, activation='gelu', normalizer='layer-norm', use_rel_pos=False, proj_drop=0., name=""
):
    for i in range(depth):
        x = SwinTransformerBlock(
                    dim                 = dim,
                    num_heads           = num_heads,
                    window_size         = window_size,
                    mlp_ratio           = mlp_ratio,
                    qkv_bias            = qkv_bias,
                    activation          = activation,
                    normalizer          = normalizer,
                    use_rel_pos         = use_rel_pos,
                    proj_drop           = proj_drop,
                    name                = f"SwinTransformerBlock/block_{i}")(x)
    return x


def ViTImageEncoder(filters=256,
                    embed_dim=768,
                    patch_size=(16, 16),
                    num_heads=12,
                    depth=12,
                    window_size=7,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    use_abs_pos=True,
                    use_rel_pos=False,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    final_activation="softmax",
                    classes=1000,
                    drop_rate=0.0):
                        
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=1024,
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

    x = PatchEmbed(patch_size, embed_dim, name="patched_embedding")(img_input)

    if use_abs_pos:
        pe_init = tf.random_normal_initializer(stddev=0.06)
        pos_embed = tf.Variable(name="pos_embedding",
                                initial_value=pe_init(shape=(1, input_shape[0] // patch_size[0], input_shape[1] // patch_size[1], embed_dim)),
                                dtype=tf.float32,
                                trainable=True)
        x = add([x, pos_embed], name="add_positional_embedding")
        
    x = BasicLayer(x,
                   dim              = embed_dim,
                   depth            = depth,
                   num_heads        = num_heads,
                   window_size      = window_size,
                   mlp_ratio        = mlp_ratio,
                   qkv_bias         = qkv_bias,
                   activation       = 'gelu',
                   normalizer       = 'layer-norm',
                   use_rel_pos      = use_rel_pos,
                   proj_drop        = drop_rate)
                        
    x = Sequential([
        Conv2D(filters=filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="valid",
               use_bias=False),
        get_normalizer_from_name('layer-norm'),
        Conv2D(filters=filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding="same",
               use_bias=False),
        get_normalizer_from_name('layer-norm'),
    ], name="neck")(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
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

    # Create model.
    model = Model(inputs, x, name='ViTImageEncoder')

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