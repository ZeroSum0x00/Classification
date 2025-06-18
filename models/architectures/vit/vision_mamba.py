"""
    Vision Mamba (Vim): Vision Backbone with Bidirectional State Space Models
    
    Overview:
        Vision Mamba (Vim) is a pure-state-space-model (SSM) based vision backbone,
        replacing self-attention with **bidirectional Mamba blocks** to achieve
        global context modeling, positional awareness, and high efficiency.
        It delivers superior accuracy to DeiT on ImageNet and downstream detection
        and segmentation, while being **2.8× faster** and using **~87% less GPU memory**
        on high-resolution images.
    
    Key Components:
        • Patch Embedding + Positional Encoding:
            - Split image into non-overlapping patches (e.g. 16×16), project to tokens.
            - Add learnable position embeddings (`Epos`) and a class token token in the *middle* for best performance (achieves 76.1% Top‑1 on ImageNet).
    
        • Vim Block:
            - **Bidirectional SSMs**:
                - Implements both forward and backward Mamba SSM layers.
                - Optionally includes a Conv1D layer before the backward SSM.
            - **Structure**:
                ```
                x' = LayerNorm(x)
                y_fwd = MambaSSM_fwd(x')
                y_bwd = Conv1D → MambaSSM_bwd(x'), // bidirectional part
                combined = y_fwd + y_bwd + x
                output = MLP(LN(combined)) + combined
                ```
    
        • Architecture:
            - Stages of stacked Vim blocks with downsampling layers.
              Example (Tiny): [Patch Embed → L blocks] where L ~ 24.
            - Overall resolution reduced via patch embedding; no explicit 2D convolutions.
    
        • Efficiency:
            - **Subquadratic runtime** and **linear memory complexity** wrt sequence length.
            - **2.8× faster** and uses **~86.8% less GPU memory** than DeiT extracting features from 1248×1248 images.
    
        • Performance:
            - Outperforms DeiT on ImageNet classification, COCO detection, and ADE20k segmentation without attention mechanisms.

    Model Parameter Comparison:
       --------------------------------------
      |     Model Name     |    Params       |
      |--------------------------------------|
      |      ViM-Base      |   8,543,720     |
       ---------------------------------------

    References:
        - Paper: “Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model”  
          https://arxiv.org/pdf/2401.09417
    
        - PyTorch implementation:
          https://github.com/hustvl/Vim
          https://github.com/state-spaces/mamba  
          https://github.com/Dao-AILab/flash-attention/tree/main/vision_mamba

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    ExtractPatches, SSM, ReduceWrapper,
)
from utils.model_processing import process_model_input, create_layer_instance, check_regularizer



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

    def __init__(
        self,
        dim,
        dt_rank,
        dim_inner,
        d_state,
        activation="silu",
        normalizer="layer-norm",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super(MambaEncoderBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.proj = Dense(
            units=self.dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.forward_conv1d = Conv1D(
            filters=self.dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.backward_conv1d = Conv1D(
            filters=self.dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.regularizer_decay,
        )
        
        self.ssm1 = SSM(
            dt_rank=self.dt_rank,
            dim_inner=self.dim_inner,
            d_state=self.d_state,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.ssm2 = SSM(
            dt_rank=self.dt_rank,
            dim_inner=self.dim_inner,
            d_state=self.d_state,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
        )
        
        self.norm_layer = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.activation = get_activation_from_name(self.activation)

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


def ViM(
    dim=256,
    dt_rank=32,
    dim_inner=256,
    d_state=256,
    patch_size=16,
    lasted_dim=768,
    num_layers=12,
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
    }

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )
    
    x = ExtractPatches(
        patch_size=patch_size,
        lasted_dim=lasted_dim,
        name="extract_patches"
    )(inputs)
    
    x = Sequential([
        Dense(
            units=dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
            
        ),
        Dropout(rate=drop_rate)
    ], name="project_patches")(x)

    for i in range(num_layers):
        x = MambaEncoderBlock(
            dim=dim,
            dt_rank=dt_rank,
            dim_inner=dim_inner,
            d_state=d_state,
            **layer_constant_dict,
            name=f"block_{i + 1}"
        )(x)

    x = ReduceWrapper(reduce_mode="mean", axis=1)(x)
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="encoder_norm")(x)

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "ViM"
    if num_layers == 12:
        model_name += "-base"
    elif num_layers == 24:
        model_name += "-large"
    elif num_layers == 32:
        model_name += "-huge"
    model_name += f"-{patch_size}"
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ViM_B16(
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
    drop_rate=0.1
):

    model = ViM(
        dim=256,
        dt_rank=32,
        dim_inner=256,
        d_state=256,
        patch_size=16,
        lasted_dim=768,
        num_layers=12,
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
