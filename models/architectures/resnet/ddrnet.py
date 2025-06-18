"""
    DDRNet: Dual-Resolution Backbone for Real-Time Dense Prediction
    
    Overview:
        DDRNet (Deep Dual-Resolution Network) is a real-time CNN backbone designed
        for high-quality **semantic segmentation** and dense prediction tasks. It
        combines a high-resolution path to preserve spatial details and a low-resolution
        path to capture semantic context, fusing them repeatedly throughout the network.
    
        Key innovations include:
            - Dual-Resolution Paths: One for fine detail, one for rich semantics
            - Fusion Modules: Bi-directional information exchange across paths
            - Deep Supervision: Auxiliary outputs to aid training and convergence
    
    Key Components:
        • Dual-Resolution Branching:
            - **High-Resolution Branch**:
                - Maintains full spatial resolution (1× downsample) for boundary and detail
            - **Low-Resolution Branch**:
                - Deeper path with larger receptive field (4× or 8× downsample)
                - Learns global semantic features
    
        • Fusion Modules:
            - At each stage, two branches exchange information:
                - Low → High: Upsample + Conv1×1
                - High → Low: Downsample + Conv3×3
            - Helps **align context and details** across scales
    
        • Residual Blocks:
            - Both branches use **ResNet-style residual blocks**
            - Typically: Bottleneck or BasicBlock (depending on DDRNet version)
    
        • Deep Supervision:
            - Auxiliary segmentation heads at intermediate stages
            - Used during training to stabilize gradient flow
    
        • Model Variants:
            - **DDRNet-23**: Most commonly used version for real-time segmentation
            - **DDRNet-39 / 23-slim**: Lighter versions for embedded deployment
    
        • Head Module:
            - Final fused features are passed through:
                - Conv1×1 → BatchNorm → Upsample → Prediction Head

    General Model Architecture:
         --------------------------------------
        |     Model Name      |    Params      |
        |--------------------------------------|
        |    DDRNet23-slim    |    7,589,256   |
        |--------------------------------------|
        |       DDRNet23      |   28,245,800   |
        |--------------------------------------|
        |       DDRNet39      |   40,170,280   |
         --------------------------------------

    References:
        - Paper: “Deep Dual-Resolution Networks for Real-Time and Accurate Semantic Segmentation of Road Scenes”  
          https://arxiv.org/abs/2101.06085
    
        - Official PyTorch implementation:  
          https://github.com/ydhongHIT/DDRNet
    
        - Variants pretrained on Cityscapes and OpenMMLab:  
          https://github.com/SegmentationBLWX/sssegmentation/tree/main/ssseg/modules/models/ddrnet

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, UpSampling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D,
    add,
)

from .resnet import basic_block, bottle_neck
from models.layers import get_activation_from_name, get_normalizer_from_name, LinearLayer
from utils.model_processing import (
    create_layer_instance, process_model_input,
    correct_pad, validate_conv_arg, check_regularizer,
    create_model_backbone,
)



def bilateral_fusion(
    low_branch,
    high_branch,
    up_size=(2, 2),
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"bilateral_fusion_block_{K.get_uid('bilateral_fusion')}"
        
    filters = high_branch.shape[-1]
    up_size = validate_conv_arg(up_size)
    regularizer_decay = check_regularizer(regularizer_decay)

    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        UpSampling2D(size=up_size, interpolation="bilinear"),
    ], name=f"{name}.high_branch.conv_block")(low_branch)
    
    x = add([high_branch, x], name=f"{name}.high_branch.fusion")

    y = high_branch
    for i in range(int(np.mean(up_size)) // 2):
        y = Sequential([
            Conv2D(
                filters=filters * 2**(i+1),
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=regularizer_decay,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation) if i != (int(np.mean(up_size)) // 2 - 1) else LinearLayer()
        ], name=f"{name}.low_branch.conv_block{i + 1}")(y)

    y = add([low_branch, y], name=f"{name}.low_branch.fusion")
    return y, x


def DDRNet23(
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
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
        "use_bias": False,
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

    # Stage 0:
    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)

    # Stage 1:
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            basic_block,
            inputs=x,
            filters=[filters, filters],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage1.final_activ")(x)

    # Stage 2:
    for i in range(num_blocks[1]):
        x = create_layer_instance(
            basic_block,
            inputs=x,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage2.final_activ")(x)

    # Stage 3:
    low_branch = x
    for i in range(num_blocks[2]):
        low_branch = create_layer_instance(
            basic_block,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage3.low_branch.block{i + 1}"
        )

    low_branch = get_activation_from_name(activation, name="stage3.low_branch.final_activ")(low_branch)

    high_branch = x
    for i in range(2):
        high_branch = create_layer_instance(
            basic_block,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage3.high_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage3.high_branch.final_activ")(high_branch)

    low_branch, high_branch = create_layer_instance(
        bilateral_fusion,
        low_branch=low_branch,
        high_branch=high_branch,
        up_size=2,
        **layer_constant_dict,
        name="stage3.bilateral_fusion"
    )

    # Stage 4:
    low_branch = get_activation_from_name(activation, name="stage4.low_branch.first_activ")(low_branch)

    for i in range(num_blocks[2]):
        low_branch = create_layer_instance(
            basic_block,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3],
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage4.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage4.high_branch.first_activ")(high_branch)

    for i in range(2):
        high_branch = create_layer_instance(
            basic_block,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage4.high_branch.block{i + 1}"
        )

    low_branch, high_branch = create_layer_instance(
        bilateral_fusion,
        low_branch=low_branch,
        high_branch=high_branch,
        up_size=4,
        **layer_constant_dict,
        name="stage4.bilateral_fusion"
    )

    # Stage 5:
    low_branch = get_activation_from_name(activation, name="stage5.low_branch.first_activ")(low_branch)

    for i in range(num_blocks[4]):
        low_branch = create_layer_instance(
            bottle_neck,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3, filters * channel_scale**4],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            use_final_activ=True,
            **layer_constant_dict,
            name=f"stage5.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage5.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[4]):
        high_branch = create_layer_instance(
            bottle_neck,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            use_final_activ=True,
            **layer_constant_dict,
            name=f"stage5.high_branch.block{i + 1}"
        )

    high_branch = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**3,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"stage5.high_branch.block{i + 2}")(high_branch)

    x = add([low_branch, high_branch], name="stage5.fusion")
    x = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**5 * final_channel_scale,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage5.merger")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model = Model(inputs=inputs, outputs=x, name="DDRNet-23-slim")
    return model


def DDRNet23_backbone(
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    return create_model_backbone(
        model_fn=DDRNet23,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DDRNet39(
    filters=64,
    num_blocks=[3, 4, 6, 3, 1],
    channel_scale=2,
    final_channel_scale=1,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
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
        "use_bias": False,
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

    # Stage 0:
    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)

    # Stage 1:
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            basic_block,
            inputs=x,
            filters=[filters, filters],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage1.final_activ")(x)

    for i in range(num_blocks[1]):
        x = create_layer_instance(
            basic_block,
            inputs=x,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage2.final_activ")(x)

    # Stage 3:
    low_branch = x
    for i in range(num_blocks[2] // 2):
        low_branch = create_layer_instance(
            basic_block,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage3.low_branch.block{i + 1}"
        )

    low_branch = get_activation_from_name(activation, name="stage3.low_branch.final_activ")(low_branch)

    high_branch = x
    for i in range(num_blocks[2] // 2):
        high_branch = create_layer_instance(
            basic_block,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage3.high_branch.block{i + 1}"
        )
        
    high_branch = get_activation_from_name(activation, name="stage3.high_branch.final_activ")(high_branch)

    low_branch, high_branch = create_layer_instance(
        bilateral_fusion,
        low_branch=low_branch,
        high_branch=high_branch,
        up_size=2,
        **layer_constant_dict,
        name="stage3.bilateral_fusion"
    )

    # Stage 4:
    low_branch = get_activation_from_name(activation, name="stage4.low_branch.first_activ")(low_branch)

    for i in range(num_blocks[2] // 2):
        low_branch = create_layer_instance(
            basic_block,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage4.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage4.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[2] // 2):
        high_branch = create_layer_instance(
            basic_block,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage4.high_branch.block{i + 1}"
        )

    low_branch, high_branch = create_layer_instance(
        bilateral_fusion,
        low_branch=low_branch,
        high_branch=high_branch,
        up_size=2,
        **layer_constant_dict,
        name="stage4.bilateral_fusion"
    )

    # Stage 5:
    low_branch = get_activation_from_name(activation, name="stage5.low_branch.first_activ")(low_branch)

    for i in range(num_blocks[3]):
        low_branch = create_layer_instance(
            basic_block,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3],
            kernel_size=(3, 3),
            strides=(2, 2) if i == 0 else (1, 1),
            residual=(i == 0),
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage5.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage5.high_branch.first_activ")(high_branch)
  
    for i in range(num_blocks[3]):
        high_branch = create_layer_instance(
            basic_block,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            use_final_activ=(i == 0),
            **layer_constant_dict,
            name=f"stage5.high_branch.block{i + 1}"
        )

    low_branch, high_branch = create_layer_instance(
        bilateral_fusion,
        low_branch=low_branch,
        high_branch=high_branch,
        up_size=4,
        **layer_constant_dict,
        name="stage5.bilateral_fusion"
    )

    # Stage 6:
    low_branch = get_activation_from_name(activation, name="stage6.low_branch.first_activ")(low_branch)

    for i in range(num_blocks[4]):
        low_branch = create_layer_instance(
            bottle_neck,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3, filters * channel_scale**4],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            use_final_activ=True,
            **layer_constant_dict,
            name=f"stage6.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage6.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[4]):
        high_branch = create_layer_instance(
            bottle_neck,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            use_final_activ=True,
            **layer_constant_dict,
            name=f"stage6.high_branch.block{i + 1}"
        )

    high_branch = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**3,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"stage6.high_branch.block{i + 2}")(high_branch)

    x = add([low_branch, high_branch], name="stage6.fusion")
    x = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**5 * final_channel_scale,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer_decay,
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage6.merger")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)

    model = Model(inputs=inputs, outputs=x, name="DDRNet-39")
    return model


def DDRNet39_backbone(
    filters,
    num_blocks,
    channel_scale=2,
    final_channel_scale=1,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    return create_model_backbone(
        model_fn=DDRNet39,
        custom_layers=custom_layers,
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DDRNet23_slim(
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
    drop_rate=0.1
) -> Model:
    
    model = DDRNet23(
        filters=32,
        num_blocks=[2, 2, 2, 2, 1],
        channel_scale=2,
        final_channel_scale=1,
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


def DDRNet23_slim_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    return create_model_backbone(
        model_fn=DDRNet23_slim,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DDRNet23_base(
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
    drop_rate=0.1
) -> Model:
    
    model = DDRNet23(
        filters=64,
        num_blocks=[2, 2, 2, 2, 1],
        channel_scale=2,
        final_channel_scale=1,
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


def DDRNet23_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    return create_model_backbone(
        model_fn=DDRNet23_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )


def DDRNet39_base(
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
    drop_rate=0.1
) -> Model:
    
    model = DDRNet39(
        filters=64,
        num_blocks=[3, 4, 6, 3, 1],
        channel_scale=2,
        final_channel_scale=1,
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


def DDRNet39_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    return create_model_backbone(
        model_fn=DDRNet39_base,
        custom_layers=custom_layers,
        inputs=inputs,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
