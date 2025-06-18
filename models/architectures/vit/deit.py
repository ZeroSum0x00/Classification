"""
    DeiT: Data-efficient Image Transformer with Knowledge Distillation
    
    Overview:
        DeiT (Data-efficient Image Transformer) is a ViT-style backbone optimized to perform
        well **without needing massive training datasets**. It achieves strong results on
        ImageNet-1k **without extra pretraining**, thanks to **a distillation mechanism** using
        a CNN teacher during training.
    
        Key innovations include:
            - Efficient transformer training on small datasets (e.g. ImageNet-1k)
            - No need for large-scale pretraining (like JFT-300M)
            - Introduction of a **Distillation Token** alongside the [CLS] token
            - Special **Distillation Loss** that transfers knowledge from CNN teachers
    
    Key Components:
        • Patch Embedding:
            - Identical to ViT: input image split into non-overlapping patches (e.g., 16×16)
            - Each patch is flattened and linearly projected to a token
    
        • Class Token + Distillation Token:
            - Two learnable tokens are prepended to patch tokens:
                - `[CLS]`: for supervised classification
                - `[DIST]`: for distilled supervision from a teacher model (e.g., ResNet-50)
    
            ```
            Tokens = [CLS] + [DIST] + PatchTokens
            ```
    
        • Position Embedding:
            - Learnable positional embeddings added to each token (including [CLS] and [DIST])
    
        • Transformer Encoder:
            - Stack of standard Transformer blocks (LN → MHSA → FFN), same as ViT
    
        • Dual Heads:
            - Two MLP heads for output:
                - Head[CLS]: predicts class label from [CLS] token
                - Head[DIST]: predicts label from distillation token
    
            During inference: **only one of the two heads is used** (usually average or [CLS])
    
        • Distillation Loss:
            - Supervised loss from [CLS] head (cross-entropy with ground truth)
            - Distillation loss from [DIST] head (cross-entropy with teacher logits)
            - Total loss = Supervised + λ × Distillation
    
        • Training Setup:
            - Teacher = CNN (e.g., RegNetY-16GF or ResNet-50)
            - Student = DeiT model
            - Trained with data augmentation (Mixup, CutMix, RandomErasing, etc.)

    Model Parameter Comparison:
       --------------------------------------
      |     Model Name     |    Params       |
      |--------------------------------------|
      |     DeiT-Tiny      |   16,619,792    |
      |--------------------|-----------------|
      |     DeiT-Small     |   36,665,936    |
      |--------------------|-----------------|
      |     DeiT-Base      |   87,375,056    |
       --------------------------------------
       
    References:
        - Paper: “Training Data-Efficient Image Transformers & Distillation through Attention”  
          https://arxiv.org/abs/2012.12877
    
        - Official PyTorch repository:
          https://github.com/facebookresearch/deit

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
    DistillationToken,
)
from utils.model_processing import process_model_input, create_layer_instance, check_regularizer



def DeiT(
    attention_block=None,
    mlp_block=None,
    num_layers=12,
    patch_size=16,
    num_heads=6,
    mlp_dim=3072,
    lasted_dim=384,
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
    
    x = DistillationToken(name="distillation_token")(x)
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
    x_head = Lambda(lambda v: v[:, 0], name="extract_predict_token")(x)
    x_dist = Lambda(lambda v: v[:, 1], name="extract_distillation_token")(x)

    if include_head:
        x_head = Sequential([
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x_head)

        x_dist = Sequential([
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_dist")(x_dist)
        
    # if training:
    #     x = x_head, x_dist
    # else:
    # x = (x_head + x_dist) / 2

    model_name = "DeiT"
    if num_heads == 3 and lasted_dim == 192:
        model_name += "-tiny"
    elif num_heads == 6 and lasted_dim == 384:
        model_name += "-small"
    elif num_heads == 12 and lasted_dim == 768:
        model_name += "-base"
        
    model = Model(inputs=inputs, outputs=[x_head, x_dist], name=model_name)
    return model


def DeiT_Ti16(
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

    model = DeiT(
        attention_block=None,
        mlp_block=None,
        num_layers=12,
        patch_size=16,
        num_heads=3,
        mlp_dim=3072,
        lasted_dim=192,
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

def DeiT_S16(
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

    model = DeiT(
        attention_block=None,
        mlp_block=None,
        num_layers=12,
        patch_size=16,
        num_heads=6,
        mlp_dim=3072,
        lasted_dim=384,
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

                
def DeiT_B16(
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

    model = DeiT(
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