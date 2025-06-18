"""
    ViTImageEncoder: Vision Transformer Backbone for Promptable Dense Representation
    
    Overview:
        ViTImageEncoder is the Vision Transformer (ViT)-based backbone used in the
        **Segment Anything Model (SAM)**. It encodes input images into dense patch-level
        representations that support **promptable segmentation** via attention-based decoding.
    
        Key innovations include:
            - Pure ViT backbone with windowed attention for scalability
            - Flexible patch embeddings that retain spatial resolution
            - Supports universal prompts (points, boxes, masks) for zero-shot segmentation
    
    Key Components:
        • Patch Embedding:
            - The input image is divided into fixed-size non-overlapping patches
            - Each patch is projected into a token using a Conv2d layer:
                - `Patch size`: typically 16×16 or 14×14 (for ViT-B, ViT-H)
                - `Embedding dim`: e.g., 768 (ViT-B), 1024 (ViT-L), 1280 (ViT-H)

        • Transformer Encoder:
            - Stack of ViT blocks: each with Multi-Head Self-Attention (MHSA) and MLP
            - For efficiency in large models, SAM uses:
                - **Windowed attention** (localized self-attention within patch regions)
                - Interleaved with **global attention blocks** every few layers
    
            - Each ViT Block:
                - LayerNorm →
                - Multi-Head Self-Attention (Windowed or Global) →
                - LayerNorm →
                - MLP (2-layer feedforward with GELU)
    
        • Positional Embedding:
            - 2D learnable position embeddings are added to each patch token
            - Preserves spatial layout essential for segmentation
    
        • Output Feature Map:
            - Final token sequence is reshaped into a 2D grid (downsampled spatial map)
            - Output shape: `[B, C, H/patch, W/patch]` → used by SAM decoder head
    
        • ViT Variants (used in SAM):
            - **ViT-B** (Base): 12 heads, 12 layers, 768-dim
            - **ViT-L** (Large): 16 heads, 24 layers, 1024-dim
            - **ViT-H** (Huge): 16 heads, 32 layers, 1280-dim
            - All pretrained on SA-1B dataset with masked autoencoding + segmentation objectives

    Model Parameter Comparison:
       -------------------------------------------------
      |        Model Name            |     Params       |
      |-------------------------------------------------|
      |     ViTImageEncoder base     |    86,689,512    |
      |-------------------------------------------------|
      |     ViTImageEncoder large    |   304,206,824    |
      |-------------------------------------------------|
      |     ViTImageEncoder huge     |   631,837,928    |
       -------------------------------------------------
    
    References:
        - Paper: “Segment Anything”  
          https://arxiv.org/abs/2304.02643
          
        - Blog/Overview:  
          https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/
          
        - Official PyTorch repository:
          https://github.com/facebookresearch/segment-anything
    
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, GlobalAveragePooling2D,
    add
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    PositionalEmbedding, MLPBlock,
)
from utils.model_processing import (
    process_model_input, correct_pad,
    validate_conv_arg, check_regularizer,
)


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
        pad = tf.constant(
            [[0, 0,],
            [0, pad_h],
            [0, pad_w],
            [0, 0]]
        )
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
        self.extractor = Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="embedding"
        )
        
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "q_size": self.q_size,
            "k_size": self.k_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        

class WindowAttention(tf.keras.layers.Layer):

    """
        Multi-head Attention block with relative position embeddings.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        use_rel_pos=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super(WindowAttention, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rel_pos = use_rel_pos
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        
    def build(self, input_shape):
        q_size = k_size = input_shape[1:3]
        
        self.qkv_projection = Dense(
            units=self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.projection = Dense(
            units=self.dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.attention = AddingDecomposedRelationPos(q_size, k_size)
        
        if self.use_rel_pos:
            self.rel_pos_h = self.add_weight(
                shape = ((2 * input_shape[1] - 1), self.head_dim),
                initializer = tf.initializers.zeros(),
                trainable = True,
                name=f'attn.relative_position_height'
            )
            
            self.rel_pos_w = self.add_weight(
                shape = ((2 * input_shape[2] - 1), self.head_dim),
                initializer = tf.initializers.zeros(),
                trainable = True,
                name=f'attn.relative_position_width'
            )
            
    def call(self, inputs, training=False):
        B_, H, W, C = inputs.shape
        qkv = self.qkv_projection(inputs, training=training)
        qkv = tf.reshape(qkv, shape=[-1, H * W, 3, self.num_heads, C // self.num_heads])        
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        qkv = tf.reshape(qkv, shape=[3, -1 , H * W, C // self.num_heads])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ tf.transpose(k, perm=[0, 2 ,1])

        if self.use_rel_pos:
            attn = self.attention(attn, q, self.rel_pos_h, self.rel_pos_w)

        attn = tf.nn.softmax(attn, axis=-1)
        x = attn @ v
        x = tf.reshape(x, shape=[-1, self.num_heads, H, W, C // self.num_heads])
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])
        x = tf.reshape(x, shape=[-1, H, W, C])
        x = self.projection(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class SwinTransformerBlock(tf.keras.layers.Layer):

    """
        Transformer blocks with support of window attention and residual propagation blocks, according to Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            activation (str or object): Activation layer.
            normalizer (str or object): Normalization layer.
            drop_rate (float, optional): Dropout rate. Default: 0.0
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        window_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        use_rel_pos=False,
        activation="gelu",
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_rate=0.,
        *args, **kwargs
    ):
        super(SwinTransformerBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        
        self.attention = WindowAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            use_rel_pos=self.use_rel_pos,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
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
        
        self.norm_layer1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.norm_layer2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
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

        
def ViTImageEncoder(
    filters=256,
    embed_dim=768,
    patch_size=(16, 16),
    num_heads=12,
    depth=12,
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    use_abs_pos=True,
    use_rel_pos=False,
    global_attn_indexes=[],
    inputs=[1024, 1024, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')
        
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
        default_size=1024,
        min_size=32,
        weights=weights
    )

    x = PatchEmbed(patch_size, embed_dim, name="patched_embedding")(inputs)

    if use_abs_pos:
        pe_init = tf.random_normal_initializer(stddev=0.06)
        pos_embed = tf.Variable(
            initial_value=pe_init(shape=(1, inputs.shape[1] // patch_size[0], inputs.shape[2] // patch_size[1], embed_dim)),
            dtype=tf.float32,
            trainable=True,
            name="pos_embedding",
        )
        
        x = add([x, pos_embed], name="add_positional_embedding")

    for i in range(depth):
        x = SwinTransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size if i not in global_attn_indexes else 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            **layer_constant_dict,
            name=f"block_{i}")(x)

    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name="neck")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "ViTImageEncoder"
    if embed_dim == 768 and num_heads == 12 and depth == 12:
        model_name += "-base"
    elif embed_dim == 1024 and num_heads == 16 and depth == 24:
        model_name += "-large"
    elif embed_dim == 1280 and num_heads == 16 and depth == 32:
        model_name += "-huge"
        
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ViTImageEncoder_B(
    inputs=[1024, 1024, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = ViTImageEncoder(
        filters=256,
        embed_dim=768,
        patch_size=(16, 16),
        num_heads=12,
        depth=12,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        global_attn_indexes=[2, 5, 8, 11],
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
        drop_rate=drop_rate
    )
    return model


def ViTImageEncoder_L(
    inputs=[1024, 1024, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = ViTImageEncoder(
        filters=256,
        embed_dim=1024,
        patch_size=(16, 16),
        num_heads=16,
        depth=24,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        global_attn_indexes=[5, 11, 17, 23],
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
        drop_rate=drop_rate
    )
    return model


def ViTImageEncoder_H(
    inputs=[1024, 1024, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
):

    model = ViTImageEncoder(
        filters=256,
        embed_dim=1280,
        patch_size=(16, 16),
        num_heads=16,
        depth=32,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        global_attn_indexes=[7, 15, 23, 31],
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
        drop_rate=drop_rate
    )
    return model
    