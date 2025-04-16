"""
  # Description:
    - The following table comparing the params of the DDRNet in Pytorch Source 
    with Tensorflow convert Source on size 224 x 224 x 3:
      
       --------------------------------------
      |     Model Name      |    Params      |
      |--------------------------------------|
      |    DDRNet23-slim    |    7,589,256   |
      |--------------------------------------|
      |       DDRNet23      |   28,245,800   |
      |--------------------------------------|
      |       DDRNet39      |   40,170,280   |
       --------------------------------------

  # Reference:
    - [Deep Dual-resolution Networks for Real-time and Accurate Semantic 
       Segmentation of Road Scenes](https://arxiv.org/pdf/2101.06085.pdf)
    - Source: https://github.com/ydhongHIT/DDRNet

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, UpSampling2D, Flatten, Dense, Dropout,
    GlobalMaxPooling2D, GlobalAveragePooling2D, add
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import create_layer_instance, process_model_input, correct_pad



def BasicBlock(
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    residual=False,
    activation="relu",
    normalizer="batch-norm",
    disable_final_activ=False,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"basic_block_{K.get_uid('basic_block')}"

    filter1, filter2 = filters
    shortcut = inputs

    x = Conv2D(
        filters=filter1,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv1"
    )(inputs)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm1")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ1")(x)

    x = Conv2D(
        filters=filter2,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv2"
    )(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm2")(x)

    if residual:
        shortcut = Conv2D(
            filters=filter2,
            kernel_size=(1, 1),
            strides=strides,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.shortcut_conv"
        )(shortcut)

        shortcut = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.shortcut_norm")(shortcut)

    x = add([x, shortcut], name=f"{name}.fusion")

    if not disable_final_activ:
        x = get_activation_from_name(activation, name=f"{name}.activ")(x)

    return x


def Bottleneck(
    inputs,
    filters,
    kernel_size,
    strides,
    residual=False,
    activation="relu",
    normalizer="batch-norm",
    disable_final_activ=False,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"bottleneck_block_{K.get_uid('bottleneck')}"

    filter1, filter2, filter3 = filters
    shortcut = inputs

    x = Conv2D(
        filters=filter1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv1"
    )(inputs)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm1")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ1")(x)

    x = Conv2D(
        filters=filter2,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv2"
    )(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm2")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ2")(x)

    x = Conv2D(
        filters=filter3,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv3"
    )(x)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm3")(x)

    if residual:
        shortcut = Conv2D(
            filters=filter3,
            kernel_size=(1, 1),
            strides=strides,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.shortcut_conv"
        )(shortcut)

        shortcut = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.shortcut_norm")(shortcut)

    x = add([x, shortcut], name=f"{name}.fusion")

    if not disable_final_activ:
        x = get_activation_from_name(activation, name=f"{name}.activ")(x)

    return x


def bilateral_fusion(
    low_branch,
    high_branch,
    up_size=2,
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
    x = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.high_branch.conv"
    )(low_branch)

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.high_branch.norm")(x)
    x = UpSampling2D(size=up_size, interpolation="bilinear", name=f"{name}.high_branch.up")(x)
    x = add([high_branch, x], name=f"{name}.high_branch.fusion")

    y = high_branch
    for i in range(up_size // 2):
        y = Conv2D(
            filters=filters * 2**(i+1),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.low_branch.conv{i + 1}"
        )(y)

        y = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.low_branch.norm{i + 1}")(y)

        if i != (up_size // 2) - 1:
            y = get_activation_from_name(activation, name=f"{name}.low_branch.activ{i + 1}")(y)

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
    pooling=None,
    activation="relu",
    normalizer=None,
    final_activation="softmax",
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
        weights=weights,
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
            kernel_regularizer=l2(regularizer_decay),
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
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)


    # Stage 1:
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            BasicBlock,
            inputs=x,
            filters=[filters, filters],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ = (i != 0),
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage1.final_activ")(x)


    # Stage 2:
    for i in range(num_blocks[1]):
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        x = create_layer_instance(
            BasicBlock,
            inputs=x,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage2.final_activ")(x)

    
    # Stage 3:
    low_branch = x
    for i in range(num_blocks[2]):
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        low_branch = create_layer_instance(
            BasicBlock,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage3.low_branch.block{i + 1}"
        )

    low_branch = get_activation_from_name(activation, name="stage3.low_branch.final_activ")(low_branch)

    high_branch = x
    for i in range(2):
        disable_final_activ = (i != 0)

        high_branch = create_layer_instance(
            BasicBlock,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
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
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        low_branch = create_layer_instance(
            BasicBlock,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3],
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage4.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage4.high_branch.first_activ")(high_branch)

    for i in range(2):
        disable_final_activ = (i != 0)

        high_branch = create_layer_instance(
            BasicBlock,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
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
            Bottleneck,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3, filters * channel_scale**4],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            **layer_constant_dict,
            name=f"stage5.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage5.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[4]):
        high_branch = create_layer_instance(
            Bottleneck,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
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
            kernel_regularizer=l2(regularizer_decay),
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
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"stage5.high_branch.block{i + 2}")(high_branch)

    outputs = add([low_branch, high_branch], name="stage5.fusion")
    outputs = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**5 * final_channel_scale,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage5.merger")(outputs)

    if include_head:
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Flatten()(outputs)
        outputs = Dropout(rate=drop_rate)(outputs)
        outputs = Dense(
            units=1 if num_classes == 2 else num_classes,
            name="predictions"
        )(outputs)
        outputs = get_activation_from_name(final_activation)(outputs)
    else:
        if pooling == "avg":
            outputs = GlobalAveragePooling2D(name="avg_pool")(outputs)
        elif pooling == "max":
            outputs = GlobalMaxPooling2D(name="max_pool")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="DDRNet-23-slim")
    
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
    custom_layers=[],
) -> Model:

    model = DDRNet23(
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DDRNet23_slim(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer="batch-norm",
    final_activation="softmax",
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
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def DDRNet23_slim_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    model = DDRNet23_slim(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DDRNet23_base(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer="batch-norm",
    final_activation="softmax",
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
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def DDRNet23_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    model = DDRNet23_base(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DDRNet39(
    filters=64,
    num_blocks=[3, 4, 6, 3, 1],
    channel_scale=2,
    final_channel_scale=1,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    final_activation="softmax",
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
        weights=weights,
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
            kernel_regularizer=l2(regularizer_decay),
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
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name="stem")(inputs)


    # Stage 1:
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            BasicBlock,
            inputs=x,
            filters=[filters, filters],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ = (i != 0),
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage1.final_activ")(x)

    for i in range(num_blocks[1]):
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        x = create_layer_instance(
            BasicBlock,
            inputs=x,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    x = get_activation_from_name(activation, name="stage2.final_activ")(x)

    
    # Stage 3:
    low_branch = x
    for i in range(num_blocks[2] // 2):
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        low_branch = create_layer_instance(
            BasicBlock,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage3.low_branch.block{i + 1}"
        )

    low_branch = get_activation_from_name(activation, name="stage3.low_branch.final_activ")(low_branch)

    high_branch = x
    for i in range(num_blocks[2] // 2):
        disable_final_activ = (i != 0)

        high_branch = create_layer_instance(
            BasicBlock,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
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
        disable_final_activ = (i != 0)

        low_branch = create_layer_instance(
            BasicBlock,
            inputs=low_branch,
            filters=[filters * channel_scale**2, filters * channel_scale**2],
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage4.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage4.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[2] // 2):
        disable_final_activ = (i != 0)

        high_branch = create_layer_instance(
            BasicBlock,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
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
        strides = (2, 2) if i == 0 else (1, 1)
        residual = (i == 0)
        disable_final_activ = (i != 0)

        low_branch = create_layer_instance(
            BasicBlock,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3],
            kernel_size=(3, 3),
            strides=strides,
            residual=residual,
            disable_final_activ=disable_final_activ,
            **layer_constant_dict,
            name=f"stage5.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage5.high_branch.first_activ")(high_branch)
  
    for i in range(num_blocks[3]):
        disable_final_activ = (i != 0)

        high_branch = create_layer_instance(
            BasicBlock,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=False,
            disable_final_activ=disable_final_activ,
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
            Bottleneck,
            inputs=low_branch,
            filters=[filters * channel_scale**3, filters * channel_scale**3, filters * channel_scale**4],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
            **layer_constant_dict,
            name=f"stage6.low_branch.block{i + 1}"
        )

    high_branch = get_activation_from_name(activation, name="stage6.high_branch.first_activ")(high_branch)

    for i in range(num_blocks[4]):
        high_branch = create_layer_instance(
            Bottleneck,
            inputs=high_branch,
            filters=[filters * channel_scale, filters * channel_scale, filters * channel_scale**2],
            kernel_size=(3, 3),
            strides=(1, 1),
            residual=True,
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
            kernel_regularizer=l2(regularizer_decay),
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
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"stage6.high_branch.block{i + 2}")(high_branch)

    outputs = add([low_branch, high_branch], name="stage6.fusion")
    outputs = Sequential([
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * channel_scale**5 * final_channel_scale,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"stage6.merger")(outputs)

    if include_head:
        outputs = GlobalAveragePooling2D()(outputs)
        outputs = Flatten()(outputs)
        outputs = Dropout(rate=drop_rate)(outputs)
        outputs = Dense(
            units=1 if num_classes == 2 else num_classes,
            name="predictions"
        )(outputs)
        outputs = get_activation_from_name(final_activation)(outputs)
    else:
        if pooling == "avg":
            outputs = GlobalAveragePooling2D(name="avg_pool")(outputs)
        elif pooling == "max":
            outputs = GlobalMaxPooling2D(name="max_pool")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="DDRNet-39")

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
    custom_layers=[],
) -> Model:

    model = DDRNet39(
        filters=filters,
        num_blocks=num_blocks,
        channel_scale=channel_scale,
        final_channel_scale=final_channel_scale,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DDRNet39_base(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer="batch-norm",
    final_activation="softmax",
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
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def DDRNet39_base_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:

    model = DDRNet39_base(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.final_activ",
        "stage2.final_activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


if __name__ == '__main__':
    model = DDRNet23_slim(include_head=False, weights=None)
    model.summary()