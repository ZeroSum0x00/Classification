"""
    Overview:
        CoaT (Co-Scale Conv-Attentional Transformer) is a hierarchical vision transformer that 
        integrates convolutional token embeddings and a novel co-scale attention mechanism. 
        It aims to bridge the performance gap between CNNs and Vision Transformers by combining 
        the local inductive bias of convolutions with the global context modeling of Transformers.

        The key innovation is the **co-scale attention**, which allows interactions between 
        multi-scale features from different resolution branches, enabling better feature fusion 
        and improved accuracy across image classification benchmarks.

    Key Characteristics:
        - Hierarchical transformer backbone with convolutional patch embedding
        - Co-scale attention enabling cross-resolution attention between different branches
        - Lightweight convolution used for token embedding and downsampling
        - Supports both lightweight (CoaT-Lite) and large-scale (CoaT) variants
        - Strong performance with improved efficiency compared to vanilla ViT

    Model Parameter Comparison:
         ----------------------------------------
        |       Model Name     |    Params       |
        |----------------------|-----------------|
        |     CoaT-lite-tiny   |   5,721,960     |
        |----------------------|-----------------|
        |     CoaT-lite-mini   |   11,011,560    |
        |----------------------|-----------------|
        |     CoaT-lite-small  |   19,838,504    |
        |----------------------|-----------------|
        |     CoaT-mini        |   9,959,436     |
        |----------------------|-----------------|
        |     CoaT-tiny        |   19,046,924    |
         ----------------------------------------

    References:
        - Paper: "CoaT: Co-Scale Conv-Attentional Image Transformers"
          https://arxiv.org/abs/2104.06399

        - Official GitHub repository:
          https://github.com/mlpc-ucsd/CoaT

        - TensorFlow/Keras implementation:
          https://github.com/leondgarse/keras_cv_attention_models/blob/main/keras_cv_attention_models/coat/coat.py
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D, Conv2D, DepthwiseConv2D, 
    Reshape, Permute, Dropout, Dense,
    concatenate, add,
)

from models.layers import (
    SplitWrapper, TransposeWrapper, ResizeWrapper, ClassToken,
    get_normalizer_from_name, get_activation_from_name,
)
from utils.model_processing import (
    process_model_input, validate_conv_arg, check_regularizer,
)



class ConvPositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=(3, 3),
        input_height=-1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super(ConvPositionalEncoding, self).__init__(*args, **kwargs)
        self.kernel_size = validate_conv_arg(kernel_size)
        self.input_height = input_height
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)

    def build(self, input_shape):
        self.pad = [
            [0, 0],
            [self.kernel_size[0] // 2, self.kernel_size[0] // 2],
            [self.kernel_size[1] // 2, self.kernel_size[1] // 2],
            [0, 0]
        ]
        self.height = self.input_height if self.input_height > 0 else int(float(input_shape[1] - 1) ** 0.5)
        self.width = (input_shape[1] - 1) // self.height
        self.channel = input_shape[-1]

        self.dconv = DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding="valid",
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.regularizer_decay,
            name=self.name and self.name + "depth_conv"
        )

    def call(self, inputs, training=False):
        cls_token, img_token = inputs[:, :1], inputs[:, 1:]
        img_token = Reshape(target_shape=[self.height, self.width, self.channel])(img_token)
        x = tf.pad(img_token, self.pad)
        x = self.dconv(x, training=training)
        x = add([x, img_token])
        x = Reshape(target_shape=[self.height * self.width, self.channel])(x)
        x = concatenate([cls_token, x], axis=1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "input_height": self.input_height,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvRelativePositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        input_height=-1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        *args, **kwargs
    ):
        super(ConvRelativePositionalEncoding, self).__init__(*args, **kwargs)
        self.head_splits = head_splits
        self.head_kernel_size = head_kernel_size
        self.input_height = input_height
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)

    def build(self, input_shape):
        query_shape = input_shape[0]
        self.height = self.input_height if self.input_height > 0 else int(float(query_shape[2] - 1) ** 0.5)
        self.width = (query_shape[2] - 1) // self.height
        self.num_heads, self.query_dim = query_shape[1], query_shape[-1]
        self.channel_splits = [ii * self.query_dim for ii in self.head_splits]

        self.dconvs = []
        self.pads = []
        for head_split, kernel_size in zip(self.head_splits, self.head_kernel_size):
            dconv = DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=(1, 1),
                padding="valid",
                depthwise_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                depthwise_regularizer=self.regularizer_decay,
                name=self.name and self.name + "depth_conv"
            )
            
            pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
            self.dconvs.append(dconv)
            self.pads.append(pad)

    def call(self, inputs, training=False):
        query, value = inputs
        img_token_q, img_token_v = query[:, :, 1:, :], value[:, :, 1:, :]

        img_token_v = tf.transpose(img_token_v, [0, 2, 1, 3])
        img_token_v = tf.reshape(img_token_v, [-1, self.height, self.width, self.num_heads * self.query_dim])
        split_values = tf.split(img_token_v, self.channel_splits, axis=-1)

        nn = [dconv(tf.pad(split_value, pad)) for split_value, dconv, pad in zip(split_values, self.dconvs, self.pads)]
        nn = concatenate(nn, axis=-1)
        conv_v_img = tf.reshape(nn, [-1, self.height * self.width, self.num_heads, self.query_dim])
        conv_v_img = tf.transpose(conv_v_img, [0, 2, 1, 3])

        EV_hat_img = img_token_q * conv_v_img
        padding = [[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 0]]
        return tf.pad(EV_hat_img, padding)

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_splits": self.head_splits,
            "head_kernel_size": self.head_kernel_size,
            "input_height": self.input_height,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def factor_attention_conv_relative_positional_encoding(
    inputs,
    shared_crpe=None,
    num_heads=8,
    qkv_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    name=None
):
    if name is None:
        name = f"relative_positional_block{K.get_uid('factor_attention_conv_relative_positional_encoding')}"

    regularizer_decay = check_regularizer(regularizer_decay)
    blocks, dim = inputs.shape[1], inputs.shape[-1]
    key_dim = dim // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)

    qkv = Dense(
        units=dim * 3,
        use_bias=qkv_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.qkv"
    )(inputs)
    
    qq, kk, vv = SplitWrapper(num_or_size_splits=3, axis=-1)(qkv)

    qq = Reshape(target_shape=[blocks, num_heads, key_dim])(qq)
    qq = TransposeWrapper(perm=[0, 2, 1, 3])(qq)
    kk = Reshape(target_shape=[blocks, num_heads, key_dim])(kk)
    kk = TransposeWrapper(perm=[0, 2, 3, 1])(kk)
    vv = Reshape(target_shape=[blocks, num_heads, key_dim])(vv)
    vv = TransposeWrapper(perm=[0, 2, 1, 3])(vv)

    # Factorized attention.
    kk = get_activation_from_name("softmax", name=f"{name}.activ")(kk)  # On `blocks` dimension
    factor_att = qq @ (kk @ vv)

    # Convolutional relative position encoding.
    crpe_layer = shared_crpe if shared_crpe is not None else ConvRelativePositionalEncoding(
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        input_height=-1,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.crpe"
    )
    crpe_out = crpe_layer([qq, vv])

    # Merge and reshape.
    nn = add([factor_att * qk_scale, crpe_out])
    nn = Permute([2, 1, 3])(nn)
    nn = Reshape(target_shape=[blocks, dim])(nn)
    nn = Dense(
        units=dim,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.projection"
    )(nn)
    return nn


def cpe_norm_crpe(
    inputs,
    shared_cpe=None,
    shared_crpe=None,
    num_heads=8,
    qkv_bias=True,
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"cpe_norm_crpe_block{K.get_uid('cpe_norm_crpe')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)

    cpe_layer = shared_cpe if shared_cpe is not None else ConvPositionalEncoding(
        kernel_size=(3, 3),
        input_height=-1,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.cpe"
    )
    cpe_out = cpe_layer(inputs)

    nn = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(cpe_out)
    
    crpe_out = factor_attention_conv_relative_positional_encoding(
        inputs=nn,
        shared_crpe=shared_crpe,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.factoratt_crpe"
    )
    
    return cpe_out, crpe_out


def res_mlp_block(
    cpe_out,
    crpe_out,
    mlp_ratio=4,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"res_mlp_block{K.get_uid('res_mlp_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
    
    if drop_rate > 0:
        crpe_out = Dropout(
            rate=drop_rate,
            noise_shape=(None, 1, 1),
            name=f"{name}.drop1"
        )(crpe_out)
        
    cpe_crpe = add([cpe_out, crpe_out])

    # MLP
    pre_mlp = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(cpe_crpe)
    
    nn = Dense(
        units=pre_mlp.shape[-1] * mlp_ratio,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.mlp1"
    )(pre_mlp)
    
    nn = get_activation_from_name(activation)(nn)
    
    nn = Dense(
        units=pre_mlp.shape[-1],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.mlp2"
    )(nn)

    if drop_rate > 0:
        nn = Dropout(
            rate=drop_rate,
            noise_shape=(None, 1, 1),
            name=f"{name}.drop2"
        )(nn)
        
    return add([cpe_crpe, nn])


def serial_block(
    inputs,
    embed_dim,
    shared_cpe=None,
    shared_crpe=None,
    num_heads=8,
    mlp_ratio=4,
    qkv_bias=True,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"serial_block{K.get_uid('serial_block')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
        
    cpe_out, crpe_out = cpe_norm_crpe(
        inputs=inputs,
        shared_cpe=shared_cpe,
        shared_crpe=shared_crpe,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.crpe"
    )
    
    out = res_mlp_block(
        cpe_out,
        crpe_out,
        mlp_ratio=mlp_ratio,
        activation=activation,
        normalizer=normalizer,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        name=f"{name}.res_mlp"
    )
    
    return out


def resample(image, target_shape, class_token=None):
    out_image = ResizeWrapper(size=target_shape, method="bilinear")(image)

    if class_token is not None:
        out_image = Reshape(target_shape=[out_image.shape[1] * out_image.shape[2], out_image.shape[-1]])(out_image)
        return concatenate([class_token, out_image], axis=1)
    else:
        return out_image


def parallel_block(
    inputs,
    shared_cpes=None,
    shared_crpes=None,
    block_heights=[],
    num_heads=8,
    mlp_ratios=[],
    qkv_bias=True,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0,
    name=None
):
    if name is None:
        name = f"parallel_block{K.get_uid('parallel_block')}"
    
    regularizer_decay = check_regularizer(regularizer_decay)
        
    # Conv-Attention.
    cpe_outs, crpe_outs, crpe_images, resample_shapes = [], [], [], []
    block_heights = block_heights[1:]
    for id, (xx, shared_cpe, shared_crpe) in enumerate(zip(inputs[1:], shared_cpes[1:], shared_crpes[1:])):
        cur_name = name + "{}.".format(id + 2)
        
        cpe_out, crpe_out = cpe_norm_crpe(
            xx,
            shared_cpe,
            shared_crpe,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            name=cur_name
        )
        
        cpe_outs.append(cpe_out)
        crpe_outs.append(crpe_out)
        height = block_heights[id] if len(block_heights) > id else int(float(crpe_out.shape[1] - 1) ** 0.5)
        width = (crpe_out.shape[1] - 1) // height
        reshaped_crpe_out = Reshape(target_shape=[height, width, crpe_out.shape[-1]])(crpe_out[:, 1:, :])
        crpe_images.append(reshaped_crpe_out)
        resample_shapes.append([height, width])
    crpe_stack = [
        crpe_outs[0] + resample(crpe_images[1], resample_shapes[0], crpe_outs[1][:, :1]) + resample(crpe_images[2], resample_shapes[0], crpe_outs[2][:, :1]),
        crpe_outs[1] + resample(crpe_images[2], resample_shapes[1], crpe_outs[2][:, :1]) + resample(crpe_images[0], resample_shapes[1], crpe_outs[0][:, :1]),
        crpe_outs[2] + resample(crpe_images[1], resample_shapes[2], crpe_outs[1][:, :1]) + resample(crpe_images[0], resample_shapes[2], crpe_outs[0][:, :1]),
    ]

    # MLP
    outs = []
    for id, (cpe_out, crpe_out, mlp_ratio) in enumerate(zip(cpe_outs, crpe_stack, mlp_ratios[1:])):
        cur_name = name + "{}.".format(id + 2)
        
        out = res_mlp_block(
            cpe_out,
            crpe_out,
            mlp_ratio=mlp_ratio,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            drop_rate=drop_rate,
            name=cur_name
        )
        
        outs.append(out)
    return inputs[:1] + outs 


def patch_embed(
    inputs,
    embed_dim,
    patch_size=2,
    input_height=-1,
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"patch_embed_block{K.get_uid('patch_embed')}"
        
    regularizer_decay = check_regularizer(regularizer_decay)
    
    if len(inputs.shape) == 3:
        input_height = input_height if input_height > 0 else int(float(inputs.shape[1]) ** 0.5)
        input_width = inputs.shape[1] // input_height
        inputs = Reshape([input_height, input_width, inputs.shape[-1]])(inputs)

    nn = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=True,
        groups=1,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=regularizer_decay,
        name=f"{name}.conv"
    )(inputs)

    block_height = nn.shape[1]
    nn = Reshape([nn.shape[1] * nn.shape[2], nn.shape[3]])(nn)  # flatten(2)
    nn = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(nn)
    return nn, block_height


def CoaT(
    num_blocks,
    embed_dims,
    mlp_ratios,
    parallel_depth,
    patch_size,
    num_heads,
    head_splits,
    head_kernel_size,
    use_shared_cpe,                 # For checking model architecture only, keep input_shape height == width if set False
    use_shared_crpe,                # For checking model architecture only, keep input_shape height == width if set False
    out_features,
    qkv_bias,
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
        
    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    # serial blocks
    x = inputs
    classfier_outs = []
    shared_cpes = []
    shared_crpes = []
    block_heights = []
    for sid, (depth, embed_dim, mlp_ratio) in enumerate(zip(num_blocks, embed_dims, mlp_ratios)):
        name = f"serial_block{sid + 1}"
        patch_size = patch_size if sid == 0 else 2
        patch_input_height = -1 if sid == 0 else block_heights[-1]
        
        x, block_height = patch_embed(
            inputs=x,
            embed_dim=embed_dim,
            patch_size=patch_size,
            input_height=patch_input_height,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps
        )
        
        block_heights.append(block_height)
        x = ClassToken(name=f"{name}.class_token")(x)
        
        shared_cpe = ConvPositionalEncoding(
            kernel_size=(3, 3),
            input_height=block_height,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            name=f"{name}.cpe"
        ) if use_shared_cpe else None
        
        shared_crpe = ConvRelativePositionalEncoding(
            head_splits=head_splits,
            head_kernel_size=head_kernel_size,
            input_height=block_height,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            name=f"{name}.crpe"
        ) if use_shared_crpe else None
        
        for bid in range(depth):
            x = serial_block(
                inputs=x,
                embed_dim=embed_dim,
                shared_cpe=shared_cpe,
                shared_crpe=shared_crpe,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                activation=activation,
                normalizer=normalizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                regularizer_decay=regularizer_decay,
                norm_eps=norm_eps,
                drop_rate=drop_rate,
                name=f"{name}.block{bid + 1}"
            )
            
        classfier_outs.append(x)
        shared_cpes.append(shared_cpe)
        shared_crpes.append(shared_crpe)
        x = x[:, 1:, :]  # remove class token

    # Parallel blocks.
    for pid in range(parallel_depth):        
        classfier_outs = parallel_block(
            inputs=classfier_outs,
            shared_cpes=shared_cpes,
            shared_crpes=shared_crpes,
            block_heights=block_heights,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            drop_rate=drop_rate,
            name=f"parallel_block{pid + 1}"
        )

    if out_features is not None:
        x = [classfier_outs[id][:, 1:, :] for id in out_features]
    elif parallel_depth == 0:
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps)(classfier_outs[-1])[:, 0]
    else:
        x = [get_normalizer_from_name(normalizer, epsilon=norm_eps)(xx)[:, :1, :] for id, xx in enumerate(classfier_outs[1:])]
        x = concatenate(x, axis=1)
        x = Permute([2, 1])(x)
        
        x = Conv1D(
            filters=1,
            kernel_size=(1,),
            strides=1,
            padding="valid",
            name="aggregate"
        )(x)
        
        x = x[:, :, 0]

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Dense(
                units=1 if num_classes == 2 else num_classes,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "CoaT"
    if embed_dims == [64, 128, 256, 320] and num_blocks == [2, 2, 2, 2]:
        model_name += "-lite-tiny"
    elif embed_dims == [64, 128, 320, 512] and num_blocks == [2, 2, 2, 2]:
        model_name += "-lite-mini"
    elif embed_dims == [64, 128, 320, 512] and num_blocks == [3, 4, 6, 3]:
        model_name += "-lite-small"
    elif embed_dims == [152, 152, 152, 152] and mlp_ratios == [4, 4, 4, 4] and parallel_depth == 6:
        model_name += "-tiny"
    elif embed_dims == [152, 216, 216, 216] and mlp_ratios == [4, 4, 4, 4] and parallel_depth == 6:
        model_name += "-mini"

    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def CoaT_ltiny(
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
) -> Model:
    
    model = CoaT(
        num_blocks=[2, 2, 2, 2],
        embed_dims=[64, 128, 256, 320],
        mlp_ratios=[8, 8, 4, 4],
        parallel_depth=0,
        patch_size=4,
        num_heads=8,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        use_shared_cpe=True,
        use_shared_crpe=True,
        out_features=None,
        qkv_bias=True,
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


def CoaT_lmini(
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
) -> Model:
    
    model = CoaT(
        num_blocks=[2, 2, 2, 2],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        parallel_depth=0,
        patch_size=4,
        num_heads=8,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        use_shared_cpe=True,
        use_shared_crpe=True,
        out_features=None,
        qkv_bias=True,
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


def CoaT_lsmall(
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
) -> Model:
    
    model = CoaT(
        num_blocks=[3, 4, 6, 3],
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        parallel_depth=0,
        patch_size=4,
        num_heads=8,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        use_shared_cpe=True,
        use_shared_crpe=True,
        out_features=None,
        qkv_bias=True,
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


def CoaT_tiny(
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
) -> Model:
    
    model = CoaT(
        num_blocks=[3, 4, 6, 3],
        embed_dims=[152, 152, 152, 152],
        mlp_ratios=[4, 4, 4, 4],
        parallel_depth=6,
        patch_size=4,
        num_heads=8,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        use_shared_cpe=True,
        use_shared_crpe=True,
        out_features=None,
        qkv_bias=True,
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


def CoaT_mini(
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
) -> Model:
    
    model = CoaT(
        num_blocks=[3, 4, 6, 3],
        embed_dims=[152, 216, 216, 216],
        mlp_ratios=[4, 4, 4, 4],
        parallel_depth=6,
        patch_size=4,
        num_heads=8,
        head_splits=[2, 3, 3],
        head_kernel_size=[3, 5, 7],
        use_shared_cpe=True,
        use_shared_crpe=True,
        out_features=None,
        qkv_bias=True,
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
