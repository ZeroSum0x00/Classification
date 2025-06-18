"""
    MLPMixer: Pure MLP-Based Backbone with Token-Channel Factorization
    
    Overview:
        MLP-Mixer is a pure MLP-based vision backbone that entirely replaces
        convolution and attention mechanisms with two simple but powerful ideas:
        **token mixing** and **channel mixing**, both performed using MLPs.
    
        It demonstrates that high-performance image classification is achievable 
        using only fully-connected layers, provided that the model can separately
        mix spatial and feature information.
    
        Key innovations include:
            - Token Mixing MLP: Mixes spatial (patch) information across tokens
            - Channel Mixing MLP: Mixes feature dimensions independently per patch
            - Separation of spatial and channel interactions leads to simplicity and scalability
    
    Key Components:
        • Patch Embedding:
            - The input image is split into non-overlapping patches (e.g., 16×16).
            - Each patch is flattened and projected using a linear layer.
            - Output is a sequence of patch embeddings (tokens) with shape `[N, C]`
              where N = number of patches, C = hidden dimension.
    
        • Mixer Block:
            - The core module, composed of two sub-blocks with residual connections:
              
              1. **Token Mixing MLP**:
                  - Operates across the patch (token) dimension
                  - Input is transposed to shape `[C, N]`  
                  - Applies MLP to each channel independently to mix spatial context
              
              2. **Channel Mixing MLP**:
                  - Operates across the channel (feature) dimension
                  - Input shape: `[N, C]`  
                  - Applies MLP to each token independently to mix channel-wise features
    
              3. **Residual Additions**:
                  - Both mixing operations are followed by residual connections and LayerNorm.
    
        • Model Structure:
            - Input → Patch Embedding → Repeated Mixer Blocks (e.g., 8–32)
            - Followed by Global Average Pooling and an MLP head for classification

        • No Attention or Convolutions:
            - Simpler than ViT and CNNs, easy to implement, and highly parallelizable.
            - Competitive performance on large datasets like ImageNet-21k or JFT-300M.

    Model Parameter Comparison:
       --------------------------------------------
      |        Model Name        |    Params       |
      |--------------------------------------------|
      |     MLPMixer-small-16    |   18,528,264    |
      |--------------------------------------------|
      |     MLPMixer-small-32    |   19,104,624    |
      |--------------------------------------------|
      |     MLPMixer-base-16     |   59,880,472    |
      |--------------------------------------------|
      |     MLPMixer-base-32     |   60,293,428    |
      |--------------------------------------------|
      |     MLPMixer-large-16    |   208,196,168   |
      |--------------------------------------------|
      |     MLPMixer-large-32    |   206,939,264   |
      |--------------------------------------------|
      |     MLPMixer-huge-14     |   432,350,952   |
       --------------------------------------------

    References:
        - Paper: “MLP-Mixer: An all-MLP Architecture for Vision”  
          https://arxiv.org/abs/2105.01601
    
        - Official implementation (Google Research):  
          https://github.com/google-research/vision_transformer
    
        - PyTorch implementation:  
          https://github.com/rishikksh20/MLP-Mixer-pytorch
          https://github.com/isaaccorley/mlp-mixer-pytorch

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling1D
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    ExtractPatches, MLPBlock,
)
from utils.model_processing import process_model_input, check_regularizer



class MixerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        tokens_dim,
        channels_dim,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
        activation="gelu",
        normalizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super(MixerBlock, self).__init__(*args, **kwargs)
        self.tokens_dim = tokens_dim
        self.channels_dim = channels_dim
        self.use_mlp_conv = use_mlp_conv
        self.use_mlp_bias = use_mlp_bias
        self.use_mlp_gated = use_mlp_gated
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.token_block = MLPBlock(
            mlp_dim=self.tokens_dim,
            out_dim=-1,
            use_conv=self.use_mlp_conv,
            use_bias=self.use_mlp_bias,
            use_gated=self.use_mlp_gated,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            drop_rate=self.drop_rate
        )
        
        self.channel_block = MLPBlock(
            self.channels_dim,
            out_dim=-1,
            use_conv=self.use_mlp_conv,
            use_bias=self.use_mlp_bias,
            use_gated=self.use_mlp_gated,
            activation=self.activation,
            normalizer=self.normalizer,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            regularizer_decay=self.regularizer_decay,
            norm_eps=self.norm_eps,
            drop_rate=self.drop_rate
        )
        
        self.layerNorm1 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
        self.layerNorm2 = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)

    def __token_mixing(self, x, training=False):
        y = self.layerNorm1(x, training=training)
        y = tf.transpose(y, perm=(0, 2, 1))  
        y = self.token_block(y)
        y = tf.transpose(y, perm=(0, 2, 1)) + x
        return y
    
    def __channel_mixing(self, x, training=False):
        y = self.layerNorm2(x, training=training)
        y = self.channel_block(y) + x
        return y

    # @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        tok = self.__token_mixing(inputs, training=training)
        output = self.__channel_mixing(tok, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "tokens_dim": self.tokens_dim,
            "channels_dim": self.channels_dim,
            "norm_eps": self.norm_eps,
            "drop_rate": self.drop_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def MLPMixer(
    patch_size,
    num_blocks,
    lasted_dim,
    tokens_dim,
    channels_dim,
    use_mlp_conv=False,
    use_mlp_bias=True,
    use_mlp_gated=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
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
    )(inputs)

    for i in range(num_blocks):
        x = MixerBlock(
            tokens_dim,
            channels_dim,
            use_mlp_conv=use_mlp_conv,
            use_mlp_bias=use_mlp_bias,
            use_mlp_gated=use_mlp_gated,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            drop_rate=drop_rate,
            name=f"stage{i + 1}"
        )(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage{i + 1}.final_norm")(x)
    
    if include_head:
        x = Sequential([
            GlobalAveragePooling1D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "MLPMixer"
    if num_blocks == 8:
        model_name += "-S"
    elif num_blocks == 12:
        model_name += "-B"
    elif num_blocks == 24:
        model_name += "-L"
    elif num_blocks == 32:
        model_name += "-H"
    model_name += f"-{patch_size}"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def MLPMixer_S16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=16,
        num_blocks=8,
        lasted_dim=512,
        tokens_dim=256,
        channels_dim=2048,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_S32(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=32,
        num_blocks=8,
        lasted_dim=512,
        tokens_dim=256,
        channels_dim=2048,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_B16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=16,
        num_blocks=12,
        lasted_dim=768,
        tokens_dim=384,
        channels_dim=3072,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_B32(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=32,
        num_blocks=12,
        lasted_dim=768,
        tokens_dim=384,
        channels_dim=3072,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_L16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=16,
        num_blocks=24,
        lasted_dim=1024,
        tokens_dim=512,
        channels_dim=4096,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_L32(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=32,
        num_blocks=24,
        lasted_dim=1024,
        tokens_dim=512,
        channels_dim=4096,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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


def MLPMixer_H14(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="gelu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
) -> Model:
    
    model = MLPMixer(
        patch_size=14,
        num_blocks=32,
        lasted_dim=1280,
        tokens_dim=640,
        channels_dim=5120,
        use_mlp_conv=False,
        use_mlp_bias=True,
        use_mlp_gated=False,
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
    