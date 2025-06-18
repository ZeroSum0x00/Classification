"""
    RepVGG: VGG-Style Backbone with Re-parameterizable Conv Layers
    
    Overview:
        RepVGG is a convolutional backbone that combines the simplicity of **VGG-style**
        architectures with the performance of modern deep CNNs. It uses a **structural
        re-parameterization** strategy: a multi-branch training-time block is converted
        into a simple **3×3 Conv + ReLU** block at inference for **fast deployment**.
    
        Key innovations include:
            - Training-time multi-branch blocks (3×3 + 1×1 + Identity)
            - Deployment-time single-path 3×3 conv → fast inference like VGG
            - No BatchNorm dependency after re-parameterization
    
    Key Components:
        • RepVGG Block (Training Mode):
            - During training, each block contains:
                - 3×3 Conv + BN
                - 1×1 Conv + BN (enhances representation)
                - Identity + BN (residual, if input/output match)
            - Outputs are summed then passed through ReLU

        • RepVGG Block (Deployment Mode):
            - All branches are **merged** into a **single 3×3 convolution kernel**
            - Result: a VGG-style plain network with only conv+ReLU → highly optimized on hardware

        • Stage-wise Architecture:
            - The network is divided into 4-5 stages with downsampling between them
            - Each stage consists of multiple RepVGG blocks
            - Stride-2 conv at stage transition for spatial downsampling

        • Re-parameterization:
            - All BN+Conv+Identity branches merged offline:
                - `Conv_fused = Conv3x3 + Conv1x1 padded + Identity as 1x1 conv`
                - `BN folded` into conv weights
            - Final model has **only Conv3x3 + ReLU** blocks
    
        • Variants:
            - **RepVGG-A0 to A3**: baseline with different widths
            - **RepVGG-B0 to B3g4**: larger models
            - **RepVGG-D2se**: SOTA performance with Squeeze-and-Excitation (SE) blocks
    
    Deployment Advantage:
        - VGG-style simple architecture (only Conv + ReLU)
        - High compatibility with TensorRT, ONNX, mobile inference
    
    General Model Architecture:
         -------------------------------------------------------------------------
        | Stage         | Layer                         | Output Shape            |
        |---------------+-------------------------------+-------------------------|
        | Input         | input_layer                   | (None, 224, 224, 3)     |
        |---------------+-------------------------------+-------------------------|
        | Stem          | RepVGGBlock (3x3, s=2)        | (None, 112, 112, 64)    |
        |---------------+-------------------------------+-------------------------|
        | Stage 1       | RepVGGBlock (3x3, s=2)        | (None, 56, 56, 160)     |
        |               | RepVGGBlock (x3)              | (None, 56, 56, 160)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 2       | RepVGGBlock (3x3, s=2)        | (None, 28, 28, 320)     |
        |               | RepVGGBlock (x3)              | (None, 28, 28, 320)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 3       | RepVGGBlock (3x3, s=2)        | (None, 14, 14, 640)     |
        |               | RepVGGBlock (x3)              | (None, 14, 14, 640)     |
        |---------------+-------------------------------+-------------------------|
        | Stage 4       | RepVGGBlock (3x3, s=2)        | (None, 7, 7, 2560)      |
        |---------------+-------------------------------+-------------------------|
        | CLS Logics    | AdaptiveAvgPooling2D          | (None, 1, 1, 2560)      |
        |               | Flatten                       | (None, 2560)            |
        |               | fc (Logics)                   | (None, 1000)            |
         -------------------------------------------------------------------------
         
    Model Parameter Comparison:
         ----------------------------------------------------------------------
        |      Model Name       |    Un-deploy params    |    Deploy params    |
        |----------------------------------------------------------------------|
        |       RepVGG-A0       |         9,132,616      |      8,309,384      |
        |----------------------------------------------------------------------|
        |       RepVGG-A1       |         14,122,088     |     12,789,864      |
        |----------------------------------------------------------------------|
        |       RepVGG-A2       |         28,253,160     |     25,499,944      |
        |----------------------------------------------------------------------|
        |       RepVGG-B0       |         15,853,160     |     14,339,048      |
        |----------------------------------------------------------------------|
        |       RepVGG-B1       |         57,483,112     |     51,829,480      |
        |----------------------------------------------------------------------|
        |      RepVGG-B1g2      |         45,850,472     |     41,360,104      |
        |----------------------------------------------------------------------|
        |      RepVGG-B1g4      |         40,034,152     |     36,125,416      |
        |----------------------------------------------------------------------|
        |       RepVGG-B2       |         89,107,432     |     80,315,112      |
        |----------------------------------------------------------------------|
        |      RepVGG-B2g2      |         70,931,432     |     63,956,712      |
        |----------------------------------------------------------------------|
        |      RepVGG-B2g4      |         61,843,432     |     55,777,512      |
        |----------------------------------------------------------------------|
        |       RepVGG-B3       |        123,185,256     |    110,960,872      |
        |----------------------------------------------------------------------|
        |      RepVGG-B3g2      |         97,011,816     |     87,404,776      |
        |----------------------------------------------------------------------|
        |      RepVGG-B3g4      |         83,925,096     |     75,626,728      |
         ----------------------------------------------------------------------

    References:
        - Paper: “RepVGG: Making VGG-style ConvNets Great Again”  
          https://arxiv.org/abs/2101.03697

        - TensorFlow/Keras implementation:
          https://github.com/hoangthang1607/RepVGG-Tensorflow-2/tree/main
          
        - PyTorch version (community):  
          https://github.com/DingXiaoH/RepVGG

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D,
)

from models.layers import (
    LinearLayer, AdaptiveAvgPooling2D,
    get_activation_from_name, get_normalizer_from_name
)
from utils.model_processing import (
    process_model_input, compute_padding,
    validate_conv_arg, check_regularizer,
    create_model_backbone,
)
from utils.logger import logger



optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


class SEBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        expansion=0.5,
        activation="relu",
        normalizer=None,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super(SEBlock, self).__init__()
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        bs = input_shape[-1]
        hidden_dim = int(bs * self.expansion)
        
        self.conv1 = Sequential([
            Conv2D(
                filters=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name(self.activation),
        ])
        
        self.conv2 = Sequential([
            Conv2D(
                filters=bs,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name("hard-sigmoid"),
        ])
        
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv1(x, training=training)        
        x = self.conv2(x, training=training)
        return x * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RepVGGBlock(tf.keras.layers.Layer):

    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 1),
        groups=1,
        use_se=False,
        activation="relu",
        normalizer="batch-norm",
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super(RepVGGBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = validate_conv_arg(kernel_size)
        self.strides = validate_conv_arg(strides)
        self.padding = padding
        self.dilation_rate = validate_conv_arg(dilation_rate)
        self.groups = groups
        self.use_se = use_se
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.deploy = deploy

        if padding.lower() not in ["same", "valid"]:
            raise ValueError(f"Invalid padding type: {padding}. Expected 'same' or 'valid'.")

        if self.padding.lower() == "valid" and not self.deploy:
            logger.warning("Using padding='valid' in training mode requires manual shape matching!")
            
    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        if self.padding.lower() == "same":
            pad_h, pad_w = compute_padding(self.kernel_size, self.dilation_rate)
            pad_1x1_h, pad_1x1_w = compute_padding((1, 1), self.dilation_rate)
        else:
            pad_h, pad_w = 0, 0
            pad_1x1_h, pad_1x1_w = 0, 0

        self.nonlinearity = get_activation_from_name(self.activation, name=f"{self.name}_nonlinearity")
        self.se_block = SEBlock(expansion=1/16) if self.use_se else LinearLayer()

        if self.deploy:
            self.rbr_reparam = Sequential([
                    ZeroPadding2D(padding=(pad_h, pad_w)),
                    Conv2D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding="valid",
                        dilation_rate=self.dilation_rate,
                        groups=self.groups,
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.regularizer_decay,
                    )
            ], name=f"{self.name}_rbr_reparam")
        else:
            if self.filters == self.in_channels and tuple(self.strides) == (1, 1):
                self.rbr_identity = get_normalizer_from_name(
                    self.normalizer,
                    epsilon=self.norm_eps,
                    name=f"{self.name}_rbr_identity"
                )

            self.rbr_dense = self.convolution_block(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=(pad_h, pad_w),
                dilation_rate=self.dilation_rate,
                groups=self.groups,
                name=f"{self.name}_rbr_dense"
            )
            self.rbr_1x1 = self.convolution_block(
                filters=self.filters,
                kernel_size=(1, 1),
                strides=self.strides,
                padding=(pad_1x1_h, pad_1x1_w),
                dilation_rate=self.dilation_rate,
                groups=self.groups,
                name=f"{self.name}_rbr_1x1"
            )

    def convolution_block(self, filters, kernel_size, strides, padding, dilation_rate=(1, 1), groups=1, name=None):
        return Sequential([
                ZeroPadding2D(padding=padding, name="reppadding"),
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    dilation_rate=dilation_rate,
                    groups=groups,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.regularizer_decay,
                    name="repconv"
                ),
                get_normalizer_from_name(
                    self.normalizer,
                    epsilon=self.norm_eps,
                    name="repbn"
                ),
        ], name=name
        )

    def _match_shape(self, x, target_shape):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        target_h, target_w = target_shape[0], target_shape[1]
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return x[:, offset_h:offset_h+target_h, offset_w:offset_w+target_w, :]

    def call(self, inputs, training=False):
        if self.deploy:
            x = self.rbr_reparam(inputs, training=training)
            x = self.se_block(x, training=training)
            return self.nonlinearity(x)

        x_dense = self.rbr_dense(inputs, training=training)
        x_1x1 = self.rbr_1x1(inputs, training=training)

        if self.padding.lower() == "valid":
            target_shape = tf.shape(x_dense)[1:3]

            x_1x1 = self._match_shape(x_1x1, target_shape)

            if hasattr(self, "rbr_identity"):
                id_out = self.rbr_identity(inputs, training=training)
                id_out = self._match_shape(id_out, target_shape)
            else:
                id_out = 0
        else:
            if hasattr(self, "rbr_identity"):
                id_out = self.rbr_identity(inputs, training=training)
            else:
                id_out = 0

        x = x_dense + x_1x1 + id_out
        x = self.se_block(x, training=training)
        return self.nonlinearity(x)

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
            )

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, Sequential):
            kernel = branch.get_layer("repconv").weights[0]
            running_mean = branch.get_layer("repbn").moving_mean
            running_var = branch.get_layer("repbn").moving_variance
            gamma = branch.get_layer("repbn").gamma
            beta = branch.get_layer("repbn").beta
            eps = branch.get_layer("repbn").epsilon
        else:
            assert isinstance(branch, BatchNormalization)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, input_dim, self.in_channels), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[1, 1, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
                )
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "use_se": self.use_se,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "deploy": self.deploy
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def RepVGG(
    num_blocks,
    width_multiplier=None,
    override_groups_map=None,
    use_se=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    assert len(width_multiplier) == 4
    override_groups_map = override_groups_map or dict()
    assert 0 not in override_groups_map

    regularizer_decay = check_regularizer(regularizer_decay)
    layer_constant_dict = {
        "use_se": use_se,
        "deploy": deploy,
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
        weights=weights,
    )

    filters = 64
    x = RepVGGBlock(
        filters=min(filters, int(filters * width_multiplier[0])),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        **layer_constant_dict,
        name="stem"
    )(inputs)

    layer_count = 1

    for i, num_block in enumerate(num_blocks):
        f = int(filters * 2**i * width_multiplier[i])

        for j in range(num_block):
            cur_groups = override_groups_map.get(layer_count, 1)
            
            x = RepVGGBlock(
                filters=f,
                kernel_size=(3, 3),
                strides=(2, 2) if j == 0 else (1, 1),
                padding="SAME",
                groups=cur_groups,
                **layer_constant_dict,
                name=f"stage{i + 1}.block{j + 1}"
            )(x)
            
            layer_count += 1

    if include_head:
        x = Sequential([
            AdaptiveAvgPooling2D(output_size=1),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model_name = "RepVGG"
    if override_groups_map == g2_map:
        g = 2
    elif override_groups_map == g4_map:
        g = 4
    else:
        g = ""
        
    if num_blocks == [2, 4, 14, 1]:
        model_name += f"-A"
        if width_multiplier == [0.75, 0.75, 0.75, 2.5]:
            i = 0
        elif width_multiplier == [1, 1, 1, 2.5]:
            i = 1
        elif width_multiplier == [1.5, 1.5, 1.5, 2.75]:
            i = 2
        else:
            i = ""
        model_name += str(i)
        
    elif num_blocks == [4, 6, 16, 1]:
        model_name += f"-B"
        if width_multiplier == [1, 1, 1, 2.5]:
            i = 0
        elif width_multiplier == [2, 2, 2, 4]:
            i = 1
        elif width_multiplier == [2.5, 2.5, 2.5, 5]:
            i = 2
        elif width_multiplier == [3, 3, 3, 5]:
            i = 3
        else:
            i = ""
        model_name += str(i)
    model_name = f"{model_name}g{g}" if g != "" else model_name
    
    model = Model(inputs=inputs, outputs=x, name=model_name)
    return model


def RepVGG_backbone(
    num_blocks,
    width_multiplier,
    override_groups_map=None,
    use_se=False,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        f"stage1.block{num_blocks[0]}",
        f"stage2.block{num_blocks[1]}",
        f"stage3.block{num_blocks[2]}",
    ]

    return create_model_backbone(
        model_fn=RepVGG,
        custom_layers=custom_layers,
        num_blocks=num_blocks,
        width_multiplier=width_multiplier,
        override_groups_map=override_groups_map,
        use_se=use_se,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def RepVGG_A0(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:

    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_A0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    return create_model_backbone(
        model_fn=RepVGG_A0,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_A1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:

    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_A1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    return create_model_backbone(
        model_fn=RepVGG_A1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_A2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_A2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    return create_model_backbone(
        model_fn=RepVGG_A2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B0(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B0,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B1,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B1g2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B1g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B1g2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B1g4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B1g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B1g4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B2g2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B2g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B2g2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B2g4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B2g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B2g4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B3(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B3,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B3g2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B3g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B3g2,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def RepVGG_B3g4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        use_se=False,
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
        drop_rate=drop_rate,
        deploy=deploy,
    )
    return model


def RepVGG_B3g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    return create_model_backbone(
        model_fn=RepVGG_B3g4,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    
def repvgg_reparameter(model: tf.keras.Model, structure, input_shape=(224, 224, 3), classes=1000, save_path=None):
    for layer, deploy_layer in zip(model.layers, structure.layers):
        if hasattr(layer, "deploy") and layer.deploy == True:
            logger.debug(f"layer {layer.name} is deployed, passing reparameter.")
            continue

        if hasattr(layer, "repvgg_convert"):
            kernel, bias = layer.repvgg_convert()
            deploy_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.Sequential) and layer.name != "classifier_head":
            assert isinstance(deploy_layer, tf.keras.Sequential)
            for sub_layer, deploy_sub_layer in zip(layer.layers, deploy_layer.layers):
                kernel, bias = sub_layer.repvgg_convert()
                deploy_sub_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    if save_path is not None:
        structure.save_weights(save_path)

    return structure
    