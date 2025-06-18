"""
    ViT: Vision Transformer – Patch-based Image Classification using Self-Attention
    
    Overview:
        ViT (Vision Transformer) is the first pure transformer-based architecture for image
        classification that matches or exceeds CNN performance when trained on large-scale
        datasets. It treats an image as a sequence of fixed-size patches (tokens), allowing
        the transformer architecture (originally for NLP) to process images directly.
    
        Key innovations include:
            - Image-to-patch tokenization: 16×16 non-overlapping patches
            - Class token ([CLS]) added to patch sequence for classification
            - Standard Transformer Encoder with Multi-Head Self-Attention (MHSA)
            - Pretrained on large datasets (e.g., JFT-300M) for strong performance
    
    Key Components:
        • Patch Embedding:
            - Image of shape (H×W×C) is split into N patches (e.g., 16×16)
            - Each patch is flattened into a vector and linearly projected (Patch→Token)
    
            ```
            Input Image (H×W×3) →
            Split into N patches (e.g., 16×16) →
            Flatten each patch → Linear Projection →
            Sequence of patch embeddings (N×D)
            ```
    
        • Class Token ([CLS]):
            - A learnable token prepended to the patch sequence
            - Used to represent the entire image for classification
    
        • Position Embedding:
            - Learnable or sinusoidal embeddings added to retain positional information
            - One positional embedding per token (including [CLS])
    
        • Transformer Encoder Blocks:
            - Each block contains:
                - Multi-Head Self-Attention (MHSA)
                - Feed-Forward Network (FFN)
                - LayerNorm and residual connections (PreNorm)
    
            ```
            For each encoder block:
            Input →
            LN → MHSA → Add →
            LN → FFN  → Add
            ```
    
        • Classification Head:
            - Final output of the [CLS] token is passed through an MLP head for classification
    
        • No Locality Bias:
            - Unlike CNNs, ViT has no built-in local inductive bias (like convolutions)
            - Relies on large data and training to learn locality and hierarchical features
    
        • Pretraining Strategy:
            - Pretrained on massive datasets like ImageNet-21k or JFT-300M
            - Fine-tuned on downstream tasks for high performance
    
        • Variants:
            - **ViT-B/16**: Base model with 16×16 patches
            - **ViT-L/16**: Larger model, more layers/heads
            - **ViT-H/14**: Huge model with smaller patch size (14×14)
    
    Architecture Summary:
        ```
        Input →
        Patch Embed (16×16) →
        + [CLS] Token →
        + Positional Encoding →
        → Transformer Encoder × L →
        → [CLS] Token Output →
        → MLP Head →
        → Class Probabilities
        ```
        
    Model Parameter Comparison:
       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     ViT-Base-16     |   86,604,520    |
      |---------------------|-----------------|
      |     ViT-Base-32     |   88,261,096    |
      |---------------------|-----------------|
      |     ViT-Large-16    |   304,424,936   |
      |---------------------|-----------------|
      |     ViT-Large-32    |   306,633,704   |
      |---------------------|-----------------|
      |     ViT-Huge-16     |   632,363,240   |
      |---------------------|-----------------|
      |     ViT-Huge-32     |   635,124,200   |
       ---------------------------------------

    References:
        - Paper: “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”  
          https://arxiv.org/abs/2010.11929
    
        - Official PyTorch repository:
          https://github.com/google-research/vision_transformer
    
        - TensorFlow/Keras implementation:
          https://github.com/faustomorales/vit-keras

        - PyTorch implementation:
          https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
          
"""

import copy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Lambda, Dense, Dropout,
    GlobalAveragePooling1D
)

from models.layers import (
    get_activation_from_name, get_normalizer_from_name,
    ExtractPatches, ClassificationToken, 
    MultiHeadSelfAttention, MLPBlock,
    PositionalEmbedding, TransformerEncoderBlock,
)
from utils.model_processing import process_model_input, create_layer_instance, check_regularizer



def ViT(
    attention_block=None,
    mlp_block=None,
    num_layers=12,
    patch_size=16,
    num_heads=12,
    mlp_dim=3072,
    lasted_dim=768,
    q_bias=True,
    kv_bias=False,
    use_attn_causal_mask=False,
    use_gated_mlp=False,
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
        "drop_rate": drop_rate,
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
    
    x = ClassificationToken(name="classification_token")(x)
    x = PositionalEmbedding(name="positional_embedding")(x)

    for i in range(num_layers):
        if attention_block is None:
            attn_clone = create_layer_instance(
                MultiHeadSelfAttention,
                num_heads=num_heads,
                num_embeds=-1,
                q_bias=q_bias,
                kv_bias=kv_bias,
                use_causal_mask=use_attn_causal_mask,
                **layer_constant_dict,
            )
        else:
            attn_clone = copy.deepcopy(attention_block)
            
        if mlp_block is None:
            mlp_clone = create_layer_instance(
                MLPBlock,
                mlp_dim=mlp_dim,
                out_dim=-1,
                use_conv=False,
                use_bias=q_bias,
                use_gated=use_gated_mlp,
                **layer_constant_dict,
            )
        else:
            mlp_clone = copy.deepcopy(mlp_block)

        x, _ = TransformerEncoderBlock(
            attention_block=attn_clone,
            mlp_block=mlp_clone,
            activation=activation,
            normalizer=None,
            norm_eps=norm_eps,
            drop_rate=drop_rate,
            name=f"block_{i + 1}"
        )(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="encoder_norm")(x)
    x = Lambda(lambda v: v[:, 0], name="extract_token")(x)

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
            
    model_name = "ViT"
    if num_layers == 12:
        model_name += "-base"
    elif num_layers == 24:
        model_name += "-large"
    elif num_layers == 32:
        model_name += "-huge"
    model_name += f"-{patch_size}"
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ViT_B16(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=12,
        patch_size=16,
        num_heads=12,
        mlp_dim=3072,
        lasted_dim=768,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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


def ViT_B32(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=12,
        patch_size=32,
        num_heads=12,
        mlp_dim=3072,
        lasted_dim=768,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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


def ViT_L16(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=24,
        patch_size=16,
        num_heads=16,
        mlp_dim=4096,
        lasted_dim=1024,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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


def ViT_L32(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=24,
        patch_size=32,
        num_heads=16,
        mlp_dim=4096,
        lasted_dim=1024,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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


def ViT_H16(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=32,
        patch_size=16,
        num_heads=16,
        mlp_dim=5120,
        lasted_dim=1280,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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


def ViT_H32(
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

    model = ViT(
        attention_block=None,
        mlp_block=None,
        num_layers=32,
        patch_size=32,
        num_heads=16,
        mlp_dim=5120,
        lasted_dim=1280,
        q_bias=True,
        kv_bias=False,
        use_attn_causal_mask=False,
        use_gated_mlp=False,
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
    