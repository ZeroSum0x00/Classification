"""
    Swin: Hierarchical Vision Transformer with Shifted Window Attention
    
    Overview:
        Swin Transformer (Shifted Window Transformer) is a hierarchical Vision Transformer
        designed for **dense prediction tasks** (e.g., detection, segmentation). It introduces
        **local windowed attention** with efficient computation and builds multi-scale
        features in a pyramid structure—making it suitable as a **general-purpose backbone**.
    
        Key innovations include:
            - Shifted Window Attention: Enables local-global context mixing with low cost
            - Hierarchical Structure: Builds multi-resolution features like CNNs
            - Patch Merging: Progressive downsampling across stages for efficiency
    
    Key Components:
        • Patch Partitioning:
            - Input image is split into non-overlapping patches (e.g., 4×4)
            - Each patch is flattened and projected to a token embedding (linear layer)

        • Swin Transformer Block:
            - Each block consists of:
                - LayerNorm
                - **Window-based Multi-Head Self-Attention (W-MSA)** or **Shifted-Windows (SW-MSA)**
                - FeedForward MLP (2-layer with GELU)
                - Residual Connections
    
            - Alternates between:
                - **W-MSA**: attention within non-overlapping windows
                - **SW-MSA**: attention within **shifted** windows → enables cross-window interaction
    
        • Patch Merging (Downsampling):
            - At each stage transition, 2×2 neighboring tokens are concatenated
            - Linear layer reduces dimensionality → doubles channels, halves resolution

        • Hierarchical Stages:
            - The network builds a **feature pyramid** similar to ResNet stages:
    
        • Positional Encoding:
            - Implicit via relative position bias in attention → flexible input sizes
    
        • Swin Variants:
            - **Swin-Tiny (T)**: lightweight (dim=96)
            - **Swin-Small (S)**, **Base (B)**, **Large (L)**: increasing capacity
            - Pretrained on ImageNet-1K/22K and adapted for segmentation/detection

    Model Parameter Comparison:
       -------------------------------------------------
      |           Model Name          |    Params       |
      |-------------------------------------------------|
      |     Swin Transformer tiny     |   28,288,354    |
      |-------------------------------------------------|
      |     Swin Transformer small    |   49,606,258    |
      |-------------------------------------------------|
      |     Swin Transformer base     |   87,768,224    |
      |-------------------------------------------------|
      |     Swin Transformer large    |  196,532,476    |
       -------------------------------------------------

    References:
        - Paper: “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows”  
          https://arxiv.org/abs/2103.14030
    
        - Official PyTorch repository:
          https://github.com/microsoft/Swin-Transformer

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Reshape, Dense, Dropout,
    GlobalAveragePooling1D
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    PositionalEmbedding, MLPBlock, DropPathV1, DropPathV2, LinearLayer,
)
from utils.model_processing import process_model_input, validate_conv_arg, check_regularizer



def window_partition(x, window_size):
    """Partition input into non-overlapping windows."""
    Wh, Ww = window_size
    B, H, W, C = x.shape
    # assert H % Wh == 0 and W % Ww == 0, "Input dimensions must be divisible by window size"
    
    x = tf.reshape(x, shape=(-1, H // Wh, Wh, W // Ww, Ww, C))
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=(-1, Wh, Ww, C))
    return windows


def window_reverse(windows, window_size, H, W, C):
    Wh, Ww = window_size
    # assert H % Wh == 0 and W % Ww == 0, "Input dimensions must be divisible by window size"
    
    x = tf.reshape(windows, shape=(-1, H // Wh, W // Ww, Wh, Ww, C))
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=(-1, H, W, C))
    return x


class PatchEmbed(tf.keras.layers.Layer):
    r""" Image to Patch Embedding

    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
        patch_size (int): Patch token size. Default: 4.
        drop_rate (float): Dropout rate. Default: 0
    """
    def __init__(
        self,
        embed_dim,
        patch_size=(4, 4),
        activation=None,
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        attn_drop_rate=0.,
        drop_rate=0.,
        *args, **kwargs
    ):
        super(PatchEmbed, self).__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.patch_size = validate_conv_arg(patch_size)
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        _, H, W, _ = input_shape

        if H % self.patch_size[0] != 0:
            self.height_pad = self.patch_size[0] - H % self.patch_size[0]
        else:
            self.height_pad = 0

        if W % self.patch_size[1] != 0:
            self.width_pad = self.patch_size[1] - W % self.patch_size[1]
        else:
            self.width_pad = 0
        
        self.proj_conv = Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )

        self.norm = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.dropout = Dropout(self.drop_rate)
            
    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        
        if self.height_pad > 0 or self.width_pad > 0:
            inputs = tf.pad(inputs, [
                [0, 0],
                [0, self.height_pad],
                [0, self.width_pad],
                [0, 0]
            ], mode='CONSTANT', constant_values=0)

        x = self.proj_conv(inputs, training=training)
        x = tf.reshape(x, [B, -1, self.embed_dim])
        x = self.norm(x, training=training)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PatchMerging(tf.keras.layers.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
    """
    
    def __init__(
        self,
        input_resolution,
        activation=None,
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_rate=0.,
        *args, **kwargs
    ):
        super(PatchMerging, self).__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.projection = Sequential([
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            Dense(
                units=2 * input_shape[-1],
                # units=2 * (input_shape[-1] // 4),
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_activation_from_name(self.activation),
            Dropout(self.drop_rate),
        ], name="projection")

    def call(self, inputs, training=False):
        H, W = self.input_resolution
        B, L, C = inputs.shape
        
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"inputs shape ({H}*{W}) are not even."

        x = tf.reshape(inputs, shape=[-1, H, W, C])

        if (H % 2 == 1) or (W % 2 == 1):
            pad = tf.constant(
                [[0,     0,],
                 [0,     0],
                 [W % 2, H % 2],
                 [0,     0]]
            )
            x = tf.pad(x, pad, mode='CONSTANT', constant_values=0)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])
        x = self.projection(x, training=training)
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            "input_resolution": self.input_resolution,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
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

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        activation="gelu",
        normalizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        attn_drop_rate=0.,
        drop_rate=0.,
        *args, **kwargs
    ):
        super(WindowAttention, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = validate_conv_arg(window_size)
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.attn_drop_rate = attn_drop_rate
        self.drop_rate = drop_rate
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer=tf.initializers.zeros(),
            trainable=True,
            name='attn.relative_position_bias_table'
        )
        
        with tf.init_scope():
            coords_h = np.arange(self.window_size[0])
            coords_w = np.arange(self.window_size[1])
            coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flatten = coords.reshape(2, -1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose([1, 2, 0])
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1).astype(np.int64)
            relative_position_tensor = tf.convert_to_tensor(relative_position_index)
            
        self.relative_position_index = self.add_weight(
            shape=relative_position_tensor.shape,
            initializer=tf.keras.initializers.Constant(relative_position_tensor),
            dtype=tf.int64,
            trainable=False,
            name="attn.relative_position_index"
        )
        
        self.qkv_projection = Dense(
            units=self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.projection = Sequential([
            Dense(
                units=self.dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name(self.activation),
            Dropout(self.drop_rate),
        ], name="projection")
        
        self.attention_dropout = Dropout(self.attn_drop_rate)
        
    def call(self, inputs, mask=None, training=False):
        B_, N, C = tf.unstack(tf.shape(inputs))
        qkv = self.qkv_projection(inputs, training=training)
        qkv = tf.reshape(qkv, shape=[-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))

        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, shape=[-1]),
        )
        
        relative_position_bias = tf.reshape(relative_position_bias, shape=[self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.nn.softmax(attn, axis=-1)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attention_dropout(attn, training=training)
        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.projection(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
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
    
    def __init__(
        self,
        dim,
        num_heads,
        input_resolution,
        window_size=(7, 7),
        shift_size=(0, 0),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        activation="gelu",
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_path_rate=0.,
        attn_drop_rate=0.,
        drop_rate=0.,
        *args, **kwargs,
    ):
        super(SwinTransformerBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = validate_conv_arg(window_size)
        self.shift_size = validate_conv_arg(shift_size)
        self.mlp_ratio = mlp_ratio

        if input_resolution[0] <= self.window_size[0] or input_resolution[1] <= self.window_size[1]:
            self.shift_size = (0, 0)
            self.window_size = (
                min(input_resolution[0], self.window_size[0]),
                min(input_resolution[1], self.window_size[1])
            )
            
        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]

        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_rate = drop_rate

    def build(self, input_shape):
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.attention = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            attn_drop_rate=self.attn_drop_rate,
            drop_rate=self.drop_rate,
        )
        
        self.mlp = MLPBlock(
            mlp_dim=mlp_hidden_dim,
            out_dim=-1,
            activation=self.activation,
            normalizer=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            drop_rate=self.drop_rate
        )
        
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else LinearLayer()
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)

        Wh, Ww = self.window_size
        Sh, Sw = self.shift_size
        if Sh > 0 or Sw > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (
                slice(0, -Wh),
                slice(-Wh, -Sh),
                slice(-Sh, None)
            )
            
            w_slices = (
                slice(0, -Ww),
                slice(-Ww, -Sw),
                slice(-Sw, None)
            )
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, shape=[-1, Wh * Ww])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(tf.not_equal(attn_mask, 0), -100.0 * tf.ones_like(attn_mask), attn_mask)
            attn_mask = tf.where(tf.equal(attn_mask, 0), tf.zeros_like(attn_mask), attn_mask)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        B = input_shape[0]
        L = input_shape[1]
        C = input_shape[2]
        H, W = self.input_resolution
        expected_L = H * W
        
        # Kiểm tra kích thước đúng
        tf.debugging.assert_equal(
            L, expected_L,
            message=f"Expected input length {expected_L} (H*W), but got {L}"
        )

        shortcut = inputs
        
        x = self.norm_layer1(inputs, training=training)
        x = tf.reshape(x, shape=[-1, H, W, C])
        
        Wh, Ww = self.window_size
        Sh, Sw = self.shift_size
        
        pad_right = (Ww - W % Ww) % Ww
        pad_bottom = (Wh - H % Wh) % Wh
        
        pad = tf.constant(
            [[0, 0],
            [0, pad_bottom],
            [0, pad_right],
            [0, 0]]
        )
        
        x = tf.pad(x, pad, mode="CONSTANT")
        shape = tf.shape(x)
        Hp, Wp = shape[1], shape[2]
        
        # cyclic shift
        if Sh > 0 or Sw > 0:
            shifted_x = tf.roll(x, shift=[-Sh, -Sw], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=[-1, Wh * Ww, C])

        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, mask=self.attn_mask, training=training)
        attn_windows = tf.reshape(attn_windows, shape=[-1, Wh, Ww, C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, C)

        # reverse cyclic shift
        if Sh > 0 or Sw > 0:
            x = tf.roll(shifted_x, shift=[Sh, Sw], axis=[1, 2])
        else:
            x = shifted_x

        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :H, :W, :]

        x = tf.reshape(x, shape=[-1, H * W, C])

        x = shortcut + self.drop_path(x, training=training)
        x = self.norm_layer2(x, training=training)
        x = self.mlp(x, training=training)
        x = x + self.drop_path(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "input_resolution": self.input_resolution,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "drop_path_rate": self.drop_path_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def Swin(
    embed_dim=96,
    patch_size=(4, 4),
    num_heads=[3, 6, 12, 24],
    depths=[2, 2, 6, 2],
    window_size=(7, 7),
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    use_abs_pos_emb=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    drop_rate=0.1
):
             
    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    patch_size = validate_conv_arg(patch_size)
    window_size = validate_conv_arg(window_size)
    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "activation": activation,
        "normalizer": normalizer,
        "kernel_initializer": kernel_initializer,
        "bias_initializer": bias_initializer,
        "regularizer_decay": regularizer_decay,
        "norm_eps": norm_eps,
        "drop_rate": drop_rate,
    }

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    x = PatchEmbed(
        embed_dim=embed_dim,
        patch_size=patch_size,
        **layer_constant_dict,
        name="patch_embedding"
    )(inputs)

    if use_abs_pos_emb:
        x = PositionalEmbedding(name="positional_embedding")(x)

    x = Dropout(rate=drop_rate, name="positional_dropout")(x)
    
    patches_resolution = [inputs.shape[1] // patch_size[0], inputs.shape[2] // patch_size[1]]
    dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

    for i, depth in enumerate(depths):
        dim = int(embed_dim * 2 ** i)
        input_resolution = (patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i))
        drop_path_prob = dpr[sum(depths[:i]):sum(depths[:i + 1])]

        for j in range(depth):
            x = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads[i],
                input_resolution=input_resolution,
                window_size=window_size,
                shift_size=(0, 0) if (j % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rate=drop_path_prob[j] if isinstance(drop_path_prob, list) else drop_path_prob,
                attn_drop_rate=attn_drop_rate,
                **layer_constant_dict,
                name=f"stage_{i + 1}.block_{j + 1}"
            )(x)
            
        if (i < len(depths) - 1):
            x = PatchMerging(
                input_resolution=input_resolution,
                **layer_constant_dict,
                name=f"stage_{i + 1}.block_{j + 2}"
            )(x)
            
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage_{i + 1}.block_{j + 1}.final_norm")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling1D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
        
    model_name = "SwinTransformer"
    if num_heads == [3, 6, 12, 24] and depths == [2, 2, 6, 2]:
        model_name += "-tiny"
    elif num_heads == [3, 6, 12, 24] and depths == [2, 2, 18, 2]:
        model_name += "-small"
    elif num_heads == [4, 8, 16, 32] and depths == [2, 2, 18, 2]:
        model_name += "-base"
    elif num_heads == [6, 12, 24, 48] and depths == [2, 2, 18, 2]:
        model_name += "-large"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def SwinT(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    attn_drop_rate=0.1,
    drop_path_rate=0.2,
    drop_rate=0.1
):

    model = Swin(
        embed_dim=96,
        patch_size=(4, 4),
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 6, 2],
        window_size=(7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        use_abs_pos_emb=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def SwinS(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    attn_drop_rate=0.1,
    drop_path_rate=0.3,
    drop_rate=0.1
):

    model = Swin(
        embed_dim=96,
        patch_size=(4, 4),
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2],
        window_size=(7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        use_abs_pos_emb=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def SwinB(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    attn_drop_rate=0.1,
    drop_path_rate=0.5,
    drop_rate=0.1
):

    model = Swin(
        embed_dim=128,
        patch_size=(4, 4),
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        window_size=(7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        use_abs_pos_emb=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def SwinL(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    attn_drop_rate=0.1,
    drop_path_rate=0.5,
    drop_rate=0.1
):

    model = Swin(
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=[6, 12, 24, 48],
        depths=[2, 2, 18, 2],
        window_size=(7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        use_abs_pos_emb=False,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model