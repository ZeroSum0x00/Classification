"""
  # Description:
    - The following table comparing the params of the Bidirectional Encoder representation from Image Transformer (BEiT)
    in Tensorflow on size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     BEiT-Base-16    |   86,567,656    |
      |---------------------|-----------------|
      |     BEiT-Base-16    |   86,567,656    |
       ---------------------------------------

  # Reference:
    - [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254v2.pdf)
    - Source: https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/beit/beit.py

"""

from __future__ import print_function
from __future__ import absolute_import

import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.initializers import RandomUniform, TruncatedNormal
from models.layers import (get_activation_from_name, get_normalizer_from_name, 
                           ClassToken, PositionalIndex, SAMModel,
                           PatchConv2DWithResampleWeights, PositionalEmbedding,
                           EnhanceSelfAttention, AttentionMLPBlock)
from utils.model_processing import _obtain_input_shape, drop_connect_rates_split



def BEiT(vocab_size=0,  # [Text model] Set value > 0 for building text model
         num_layers=12,
         patch_size=16,
         num_heads=12,
         embed_dim=768,
         use_patch_bias=True,  # False for MetaTransFormer, True for others
         use_pre_norm=False,  # True for MetaTransFormer, False for others
         attn_key_dim=0,  # [Attention args]
         attn_qv_bias=True,  # Default False for Vit, True for Beit, if True and attn_qkv_bias being False, will add BiasLayer for query and key.
         attn_qkv_bias=False,  # True for Vit, False for Beit, if True, will just use bias in qkv_dense, and set qv_bias False.
         attn_return_weight=True,
         attn_return_bias=True,
         attn_layer_scale=0.1,  # 0 for Vit, 0.1 for Beit, if > 0 will use `layer_scale` on block output
         attn_dropout=0,
         use_abs_pos_emb=False,  # [Pos emb args] True for Vit, False for Beit, whether use abcolute positional embedding or relative one in attention blocks
         use_abs_pos_emb_on_cls_token=True,  # [Pos emb args] False for FlexiViT, no_embed_class in timm. If use_abs_pos_emb is True, whether apply pos_emb on cls_token.
         use_rot_pos_emb=False,  # [Pos emb args] True for EVA02, False for others
         mlp_ratio=4,
         use_gated_mlp=False,  # [MLP args] True for DINOv2 and EVA02
         use_mlp_norm=False,  # [MLP args] True for EVA02 base and large, False for others.
         use_mean_pooling_head=True,  # [Head args] False for Vit, True for Beit, whether use use mean output or `class_token` output
         use_cat_head=False,  # [Head args] True for DINOv2
         max_block_size=77,  # [Text model] max block size, works only if vocab_size > 0
         text_positional_dropout=0,  # [Text model] dropout for text model embedding layers
         text_use_positional_embedding=True,  # [Text model] boolean value if use Embedding positional layer after inputs
         include_top=True, # [Text model] boolean value if include top output Dense layer, True for using output channles == vocab_size
         weights='imagenet',
         input_tensor=None, 
         input_shape=None, # [Common args] Not taking effect for text model
         pooling=None,
         activation='gelu',
         final_activation="softmax",
         classes=1000,  # For text model, equals to vocab_size if include_top is True
         sam_rho=0.0,
         norm_eps=1e-6,  # 1e-5 for ViT clip models, 1e-6 for others
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
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        if vocab_size > 0:
            img_input = Input([None], dtype="int64")
        else:
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
        
    if vocab_size > 0:
        """Text inputs"""
        tok_emb = Embedding(vocab_size, embed_dim, name="embed_tokens")(img_input)

        if text_use_positional_embedding:
            pos_idx = PositionalIndex(block_size=max_block_size, name="pos_idx")(img_input)
            pos_emb = Embedding(max_block_size, embed_dim, name="wpe")(pos_idx)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        if text_positional_dropout:
            x = Dropout(text_positional_dropout)(x)
            
        patch_height = -1
        num_classes = vocab_size
    else:
        """ forward_embeddings """
        kernel_initializer = RandomUniform(minval=-1 / (patch_size**0.5), maxval=1 / (patch_size**0.5))
        x = PatchConv2DWithResampleWeights(embed_dim, 
                                            patch_size, 
                                            strides=patch_size, 
                                            padding="valid", 
                                            kernel_initializer=kernel_initializer, 
                                            use_bias=use_patch_bias, 
                                            name="stem_conv")(img_input)
        patch_height = x.shape[1]
        x = Reshape([-1, x.shape[-1]])(x)

        """ Positional embedding """
        if use_abs_pos_emb and use_abs_pos_emb_on_cls_token:  # ViT / EvaLarge / EvaGiant / DINOv2
            x = ClassToken(name="cls_token")(x)
            x = PositionalEmbedding(name="positional_embedding")(x)
        elif use_abs_pos_emb:  # FlexiViT models
            x = PositionalEmbedding(name="positional_embedding")(x)
            x = ClassToken(name="cls_token")(x)
        else:  # Beit and BeitV2
            x = ClassToken(name="cls_token")(x)

        if use_pre_norm:
            x = get_normalizer_from_name('layer-norm', axis=-1, epsilon=norm_eps, name="pre_ln")(x)

    drop_connect_rates = drop_connect_rates_split([num_layers], 0.0, drop_rate)[0]

    for id in range(num_layers):
        block_drop_rate = drop_connect_rates[id]
        attent_layer = EnhanceSelfAttention(num_heads=num_heads,
                                            key_dim=attn_key_dim,
                                            attn_height=patch_height,
                                            qv_bias=attn_qv_bias,
                                            qkv_bias=attn_qkv_bias,
                                            return_weight=attn_return_weight,
                                            return_bias=attn_return_bias,
                                            pos_emb=not use_abs_pos_emb,
                                            rotate_pos_emb=use_rot_pos_emb,
                                            text_max_block_size=max_block_size if vocab_size > 0 else 0,
                                            attn_dropout=attn_dropout)

        x = AttentionMLPBlock(attent_layer,
                               mlp_ratio=mlp_ratio, 
                               layer_scale=attn_layer_scale, 
                               use_gated_mlp=use_gated_mlp,
                               activation=activation,
                               normalizer='layer-norm',
                               use_mlp_norm=use_mlp_norm, 
                               norm_eps=norm_eps,
                               drop_rate=block_drop_rate,
                               drop_prob=0.0,
                               name=f"AttentionMLP_block_{id + 1}")(x)

    """ Head """
    if vocab_size > 0:  # Text model
        x = get_normalizer_from_name('layer-norm', axis=-1, epsilon=norm_eps, name="out_ln")(x)
    elif use_cat_head:  # DINOv2
        x = get_normalizer_from_name('layer-norm', axis=-1, epsilon=norm_eps, name="out_ln")(x)
        x = tf.concat([x[:, 0], tf.reduce_mean(x[:, 1:, :], axis=1)], axis=-1)
    elif use_mean_pooling_head:
        x = tf.reduce_mean(x[:, 1:, :], axis=1)
        x = get_normalizer_from_name('layer-norm', axis=-1, epsilon=norm_eps, name="out_ln")(x)
    else:  # FlexiViT
        x = get_normalizer_from_name('layer-norm', axis=-1, epsilon=norm_eps, name="out_ln")(x)
        x = x[:, 0]

    if include_top:
        x = Dense(1 if classes == 2 else classes, 
                   kernel_initializer=TruncatedNormal(stddev=0.02), 
                   bias_initializer=TruncatedNormal(stddev=0.02),
                   name='predictions')(x)
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

    def __build_model(inputs, outputs, sam_rho, name):
        if sam_rho != 0:
            return SAMModel(inputs, x, name=name + '_SAM')
        else:
            return Model(inputs, x, name=name)
            
    model = __build_model(inputs, x, sam_rho, name='BEiT-Base-16')

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


def BEiTBase16(include_top=True, 
               weights='imagenet',
               input_tensor=None, 
               input_shape=None,
               pooling=None,
               final_activation="softmax",
               classes=1000,
               sam_rho=0.0,
               norm_eps=1e-6,
               drop_rate=0.1):

    model = BEiT(vocab_size=0,
                 num_layers=12,
                 patch_size=16,
                 num_heads=12,
                 embed_dim=768,
                 use_patch_bias=True,
                 use_pre_norm=False,
                 attn_key_dim=0,
                 attn_qv_bias=True,
                 attn_qkv_bias=False,
                 attn_return_weight=True,
                 attn_return_bias=True,
                 attn_layer_scale=0.1,
                 attn_dropout=0,
                 use_abs_pos_emb=False,
                 use_abs_pos_emb_on_cls_token=True,
                 use_rot_pos_emb=False,
                 mlp_ratio=4,
                 use_gated_mlp=False,
                 use_mlp_norm=False,
                 use_mean_pooling_head=True,
                 use_cat_head=False,
                 max_block_size=77,
                 text_positional_dropout=0,
                 text_use_positional_embedding=True,
                 include_top=include_top,
                 weights=weights,
                 input_tensor=input_tensor, 
                 input_shape=input_shape,
                 pooling=pooling,
                 activation='gelu',
                 final_activation=final_activation,
                 classes=classes,
                 sam_rho=sam_rho,
                 norm_eps=norm_eps,
                 drop_rate=drop_rate)
    return model


def BEiTLarge16(include_top=True, 
                weights='imagenet',
                input_tensor=None, 
                input_shape=None,
                pooling=None,
                final_activation="softmax",
                classes=1000,
                sam_rho=0.0,
                norm_eps=1e-6,
                drop_rate=0.1):

    model = BEiT(vocab_size=0,
                 num_layers=24,
                 patch_size=16,
                 num_heads=16,
                 embed_dim=1024,
                 use_patch_bias=True,
                 use_pre_norm=False,
                 attn_key_dim=0,
                 attn_qv_bias=True,
                 attn_qkv_bias=False,
                 attn_return_weight=True,
                 attn_return_bias=True,
                 attn_layer_scale=1e-05,
                 attn_dropout=0,
                 use_abs_pos_emb=False,
                 use_abs_pos_emb_on_cls_token=True,
                 use_rot_pos_emb=False,
                 mlp_ratio=4,
                 use_gated_mlp=False,
                 use_mlp_norm=False,
                 use_mean_pooling_head=True,
                 use_cat_head=False,
                 max_block_size=77,
                 text_positional_dropout=0,
                 text_use_positional_embedding=True,
                 include_top=include_top,
                 weights=weights,
                 input_tensor=input_tensor, 
                 input_shape=input_shape,
                 pooling=pooling,
                 activation='gelu',
                 final_activation=final_activation,
                 classes=classes,
                 sam_rho=sam_rho,
                 norm_eps=norm_eps,
                 drop_rate=drop_rate)
    return model