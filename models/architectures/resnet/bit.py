"""
    BiT: Big Transfer Backbone using Pretrained ResNet for Universal Representation
    
    Overview:
        BiT (Big Transfer) is a large-scale pretraining framework based on ResNet
        architectures, designed to transfer well to diverse downstream tasks
        (classification, detection, segmentation, etc.). It emphasizes **pretraining
        on massive datasets** (e.g., JFT-300M) and then fine-tuning with minimal task-specific changes.
    
        Key innovations include:
            - Scalable ResNet architectures (BiT-S, BiT-M, BiT-L)
            - Pretraining on huge, weakly-labeled datasets
            - Minimal fine-tuning: 1 training run can solve many tasks
            - Frozen early layers + linear classifier achieves strong performance
    
    Key Components:
        • Scalable ResNet Backbone:
            - Uses standard ResNet variants: ResNet-50, 101, 152, 200
            - Trained with minimal tweaks: BatchNorm, ReLU, no tricks
            - Named as:
                - **BiT-S**: Trained from scratch on ImageNet-21k
                - **BiT-M**: Pretrained on JFT-300M
                - **BiT-L**: Larger ResNet with more capacity
    
        • Pretraining Strategy:
            - Trained on massive datasets using supervised classification objective
            - Emphasis on **clean design** over architectural novelty
            - Large-scale training + weight decay + strong augmentations (e.g., RandAugment)
    
        • Transfer Learning Pipeline:
            - Pretrained BiT model is **frozen or partially fine-tuned**
            - Downstream tasks can be solved with:
                - Linear head (BiT-L ⭢ SOTA with only logistic regression)
                - Fine-tuned fully or partially

        • Simplicity Principle:
            - No custom layers (e.g., attention, gating)
            - Standard CNN but scaled up and trained properly
            - Focuses on *data and scale* instead of model complexity

    General Model Architecture:
         -------------------------------------------------------------------------
        | Stage         | Layer                         | Output Shape            |
        |---------------+-------------------------------+-------------------------|
        | Input         | input_layer                   | (None, 224, 224, 3)     |
        |---------------+-------------------------------+-------------------------|
        | Stem          | PaddingFromKernelSize (7x7)   | (None, 225, 225, 3)     |
        |               | StandardizedConv2D (7x7, s=2) | (None, 112, 112, 64)    |
        |               | StandardizedConv2D (7x7, s=2) | (None, 114, 114, 64)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 1       | MaxPooling2D (3x3, s=2)       | (None, 56, 56, 64)      |
        |               | bottleneck_unit_v2 (x3)       | (None, 56, 56, 256)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 2       | bottleneck_unit_v2 (s=2)      | (None, 28, 28, 512)     |
        |               | bottleneck_unit_v2 (x3)       | (None, 28, 28, 512)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 3       | bottleneck_unit_v2 (s=2)      | (None, 14, 14, 1024)    |
        |               | bottleneck_unit_v2 (x5)       | (None, 14, 14, 1024)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 4       | bottleneck_unit_v2 (s=2)      | (None, 7, 7, 2048)      |
        |               | bottleneck_unit_v2 (x2)       | (None, 7, 7, 2048)      |
        |---------------+-------------------------------+-------------------------|
        | CLS Logics    | GlobalAveragePooling2D        | (None, 2048)            |
        |               | fc (Logics)                   | (None, 1000)            |
         -------------------------------------------------------------------------
         
    Model Parameter Comparison:
         -------------------------------------
        |     Model Name    |    Params       |
        |-------------------|-----------------|
        |     BiT_R50x1     |   25,549,352    |
        |-------------------|-----------------|
        |     BiT_R101x1    |   44,541,480    |
        |-------------------|-----------------|
        |     BiT_R50x3     |   217,319,080   |
        |-------------------|-----------------|
        |     BiT_R101x3    |   387,934,888   |
        |-------------------|-----------------|
        |     BiT_R152x4    |   936,533,224   |
         -------------------------------------

    
    References:
        - Paper: “Big Transfer (BiT): General Visual Representation Learning”  
          https://arxiv.org/abs/1912.11370
    
        - Official code (TensorFlow + JAX):  
          https://github.com/google-research/big_transfer

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout,
    MaxPooling2D, GlobalAveragePooling2D,
    add,
)

from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import process_model_input, check_regularizer, create_model_backbone



class PaddingFromKernelSize(tf.keras.layers.Layer):
    """Layer that adds padding to an image taking into a given kernel size."""

    def __init__(self, kernel_size, **kwargs):
        super(PaddingFromKernelSize, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            pad_total = kernel_size - 1
        else:
            pad_total = kernel_size[0] - 1
            
        self._pad_beg = pad_total // 2
        self._pad_end = pad_total - self._pad_beg

    def call(self, x):
        padding = [[0,                         0],
                   [self._pad_beg, self._pad_end],
                   [self._pad_beg, self._pad_end],
                   [0,                         0]]
        return tf.pad(x, padding)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class StandardizedConv2D(Conv2D):
    """
    Chuẩn hóa kernel trước khi tích chập.
    Dựa trên paper: https://arxiv.org/abs/1903.10520
    """
    def call(self, inputs):
        # Tính mean và variance theo chiều [0, 1, 2] (H, W, in_channels)
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        # Chuẩn hóa kernel
        standardized_kernel = (self.kernel - mean) / tf.sqrt(var + 1e-10)

        # Áp dụng phép tích chập
        outputs = tf.nn.conv2d(
            inputs,
            standardized_kernel,
            strides=self.strides,
            padding=self.padding.upper(),
            dilations=self.dilation_rate,
            data_format='NHWC'
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

      
def bottleneck_unit_v2(
    inputs,
    filters,
    kernel_size=(3, 3),
    strides=(2, 2),
    use_bias=False,
    activation="relu",
    normalizer="group-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    shortcut = inputs

    x = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.pre_norm")(inputs)

    if (max(strides) > 1) or (4 * filters != inputs.shape[-1]):
        shortcut = StandardizedConv2D(
            filters=4 * filters,
            kernel_size=(1, 1),
            strides=strides,
            use_bias=use_bias,
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
            name=f"{name}.shortcut"
        )(x)

    x = Sequential([
        StandardizedConv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        PaddingFromKernelSize(kernel_size=(3, 3)),
        StandardizedConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        StandardizedConv2D(
            filters=4 * filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=use_bias,
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
    ], name=f"{name}.conv_block")(x)

    return add([x, shortcut], name=f"{name}.add")


def ResnetV2(
    filters,
    num_blocks,
    channel_scale,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
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

    filters = filters if isinstance(filters, (tuple, list)) else [filters * channel_scale**i for i in range(len(num_blocks))]
    
    x = Sequential([
        PaddingFromKernelSize(kernel_size=(7, 7)),
        StandardizedConv2D(
            filters=filters[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            use_bias=False,
        ),
        PaddingFromKernelSize(kernel_size=(3, 3)),
    ], name="stem")(inputs)
    
    for i, num_block in enumerate(num_blocks):
        for j in range(num_block):
            if i == 0 and j == 0:
                x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f"stage{i + 1}.block{j + 1}")(x)
            else:
                x = bottleneck_unit_v2(
                    inputs=x,
                    filters=filters[i],
                    kernel_size=(3, 3),
                    strides=(2, 2) if (i != 0 and j == 0) else (1, 1),
                    use_bias=False,
                    **layer_constant_dict,
                    name=f"stage{i + 1}.block{j + 1}"
                )

    x = Sequential([
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage{i + 2}.block{j + 2}")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "BiT"
    if num_blocks == (4, 4, 6, 3):
        model_name += f"-R50x"
    elif num_blocks == (4, 4, 23, 3):
        model_name += f"-R101x"
    elif num_blocks == (4, 8, 36, 3):
        model_name += f"-R152x"
    model_name += str(filters[0] // 64)
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def ResnetV2_backbone(
    filters,
    num_blocks,
    channel_scale,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        f"stage1.block{num_blocks[0]}.add",
        f"stage2.block{num_blocks[1]}.add",
        f"stage3.block{num_blocks[2]}.add",
    ]

    return create_model_backbone(
        model_fn=ResnetV2,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def BiT_R50x1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = ResnetV2(
        filters=64,
        num_blocks=(4, 4, 6, 3),
        channel_scale=2,
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


def BiT_R50x1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.add",
        "stage2.block4.add",
        "stage3.block6.add",
    ]

    return create_model_backbone(
        model_fn=BiT_R50x1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def BiT_R50x3(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = ResnetV2(
        filters=192,
        num_blocks=(4, 4, 6, 3),
        channel_scale=2,
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


def BiT_R50x3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.add",
        "stage2.block4.add",
        "stage3.block6.add",
    ]

    return create_model_backbone(
        model_fn=BiT_R50x3,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def BiT_R101x1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = ResnetV2(
        filters=64,
        num_blocks=(4, 4, 23, 3),
        channel_scale=2,
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


def BiT_R101x1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.add",
        "stage2.block4.add",
        "stage3.block23.add",
    ]

    return create_model_backbone(
        model_fn=BiT_R101x1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def BiT_R101x3(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = ResnetV2(
        filters=192,
        num_blocks=(4, 4, 23, 3),
        channel_scale=2,
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


def BiT_R101x3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.add",
        "stage2.block4.add",
        "stage3.block23.add",
    ]

    return create_model_backbone(
        model_fn=BiT_R101x3,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def BiT_R152x4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = ResnetV2(
        filters=256,
        num_blocks=(4, 8, 36, 3),
        channel_scale=2,
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

def BiT_R152x4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="group-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4.add",
        "stage2.block8.add",
        "stage3.block36.add",
    ]

    return create_model_backbone(
        model_fn=BiT_R152x4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    