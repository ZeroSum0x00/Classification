"""
  # Description:
    - The following table comparing the params of the EfficientNet in Tensorflow on 
    size 224 x 224 x 3:

       ------------------------------------------
      |     Model Name         |     Params      |
      |------------------------------------------|
      |     EfficientNet-B0    |     5,330,564   |
      |------------------------|-----------------|
      |     EfficientNet-B1    |     7,856,232   |
      |------------------------|-----------------|
      |     EfficientNet-B2    |     9,177,562   |
      |------------------------|-----------------|
      |     EfficientNet-B3    |    12,320,528   |
      |------------------------|-----------------|
      |     EfficientNet-B4    |    19,466,816   |
      |------------------------|-----------------|
      |     EfficientNet-B5    |    30,562,520   |
      |------------------------|-----------------|
      |     EfficientNet-B6    |    43,265,136   |
      |------------------------|-----------------|
      |     EfficientNet-B7    |    66,658,680   |
       ------------------------------------------

  # Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946v5.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
              https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

"""

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, ZeroPadding2D, DepthwiseConv2D,
    Reshape, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,
    multiply, add
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, correct_pad


DEFAULT_BLOCKS_ARGS = [
    {"filters_in": 32, "filters_out": 16, "kernel_size": 3, "strides": 1,
     "expand_ratio": 1, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 1},
    {"filters_in": 16, "filters_out": 24, "kernel_size": 3, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 2},
    {"filters_in": 24, "filters_out": 40, "kernel_size": 5, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 2},
    {"filters_in": 40, "filters_out": 80, "kernel_size": 3, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 3},
    {"filters_in": 80, "filters_out": 112, "kernel_size": 5, "strides": 1,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 3},
    {"filters_in": 112, "filters_out": 192, "kernel_size": 5, "strides": 2,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 4},
    {"filters_in": 192, "filters_out": 320, "kernel_size": 3, "strides": 1,
     "expand_ratio": 6, "squeeze_ratio": 0.25, "residual_connection": True, "repeats": 1},
]


def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def EfficientBlock(
    inputs,
    filters,
    kernel_size,
    strides,
    expand_ratio=1,
    squeeze_ratio=0.,
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    residual_connection=True,
    drop_rate=0.,
    name=None
):
    if name is None:
        name = f"efficient_block_{K.get_uid('efficient_block')}"

    f1, f2 = filters
    f = f1 * expand_ratio
    
    if expand_ratio != 1:
        x = Conv2D(
            filters=f,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.expand.conv"
        )(inputs)
        
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.expand.norm")(x)
        x = get_activation_from_name(activation, name=f"{name}.expand.activ")(x)
    else:
        x = inputs

    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(x, kernel_size), name=f"{name}.dwconv.padding")(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"

    x = DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=l2(regularizer_decay),
        name=f"{name}.dwconv.conv"
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.dwconv.norm")(x)
    x = get_activation_from_name(activation, name=f"{name}.dwconv.activ")(x)

    if 0 < squeeze_ratio <= 1:
        filters_squeeze = max(1, int(f1 * squeeze_ratio))
        squeeze = GlobalAveragePooling2D(name=f"{name}.squeeze.squeeze")(x)
        squeeze = Reshape((1, 1, f), name=f"{name}.squeeze.reshape")(squeeze)

        squeeze = Conv2D(
            filters=filters_squeeze,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.squeeze.reduce"
        )(squeeze)
        
        squeeze = Conv2D(
            filters=f,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="sigmoid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"{name}.squeeze.conv",
        )(squeeze)
        
        x = multiply([x, squeeze], name=f"{name}.squeeze.excite")
    
    x = Conv2D(
        filters=f2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.project.conv",
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.project.norm")(x)

    if (residual_connection is True) and (strides == 1) and (f1 == f2):
        if drop_rate > 0:
            x = Dropout(rate=drop_rate, noise_shape=(None, 1, 1, 1), name=f"{name}.drop")(x)
        x = add([x, inputs], name=f"{name}.add")

    return x


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    depth_divisor=8,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights
    )

    x = Sequential([
        ZeroPadding2D(padding=correct_pad(inputs, 3)),
        Conv2D(
            filters=round_filters(32, width_coefficient, depth_divisor),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name("batch-norm", epsilon=norm_eps),
        get_activation_from_name("swish"),
    ], name="stem")(inputs)
    
    b = 0
    blocks = float(sum(args["repeats"] for args in blocks_args))
    for idx, args in enumerate(blocks_args):
        filters_in = round_filters(args["filters_in"], width_coefficient, depth_divisor)
        filters_out = round_filters(args["filters_out"], width_coefficient, depth_divisor)
        kernel_size = args["kernel_size"]
        strides = args["strides"]
        expand_ratio = args["expand_ratio"]
        squeeze_ratio = args["squeeze_ratio"]
        residual_connection = args["residual_connection"]
        repeats = args["repeats"]

        for i in range(round_repeats(repeats, depth_coefficient)):
            if i > 0:
                strides = 1
                filters_in = filters_out

            x = EfficientBlock(
                inputs=x,
                filters=[filters_in, filters_out],
                kernel_size=kernel_size,
                strides=strides,
                expand_ratio=expand_ratio,
                squeeze_ratio=squeeze_ratio,
                activation=activation,
                normalizer=normalizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                regularizer_decay=regularizer_decay,
                norm_eps=norm_eps,
                residual_connection=residual_connection,
                drop_rate=drop_connect_rate * b / blocks,
                name=f"stage{idx + 1}.block{i + 1}"
            )
            b += 1

    x = Sequential([
        Conv2D(
            round_filters(1280, width_coefficient, depth_divisor),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name("batch-norm", epsilon=norm_eps),
        get_activation_from_name("swish"),
    ], name=f"stage{idx + 1}.block{i + 2}")(x)

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="global_avgpool")(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D(name="global_maxpool")(x)

    # Create model.
    if (width_coefficient == 1.0) and (depth_coefficient == 1.0):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B0")
    elif (width_coefficient == 1.0) and (depth_coefficient == 1.1):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B1")
    elif (width_coefficient == 1.1) and (depth_coefficient == 1.2):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B2")
    elif (width_coefficient == 1.2) and (depth_coefficient == 1.4):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B3")
    elif (width_coefficient == 1.4) and (depth_coefficient == 1.8):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B4")
    elif (width_coefficient == 1.6) and (depth_coefficient == 2.2):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B5")
    elif (width_coefficient == 1.8) and (depth_coefficient == 2.6):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B6")
    elif (width_coefficient == 2.0) and (depth_coefficient == 3.1):
        model = Model(inputs=inputs, outputs=x, name="EfficientNet-B7")
    else:
        model = Model(inputs=inputs, outputs=x, name="EfficientNet")

    return model


def EfficientNet_backbone(
    width_coefficient,
    depth_coefficient,
    blocks_args=DEFAULT_BLOCKS_ARGS,
    depth_divisor=8,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNet(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        blocks_args=blocks_args,
        depth_divisor=depth_divisor,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB0(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB0(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.2,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB2(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.3,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB3(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.3,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB3(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def EfficientNetB4(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.4,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.4,
        depth_coefficient=1.8,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB4(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def EfficientNetB5(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.4,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.6,
        depth_coefficient=2.2,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB5_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB5(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB6(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=1.8,
        depth_coefficient=2.6,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB6_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[],
) -> Model:
    
    model = EfficientNetB6(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def EfficientNetB7(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="swish",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    drop_connect_rate=0.2
) -> Model:
    
    model = EfficientNet(
        width_coefficient=2.0,
        depth_coefficient=3.1,
        blocks_args=DEFAULT_BLOCKS_ARGS,
        depth_divisor=8,
        inputs=inputs,
        include_head=include_head,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
        drop_connect_rate=drop_connect_rate
    )
    return model


def EfficientNetB7_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = EfficientNetB7(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage2.block1.expand.activ",
        "stage3.block1.expand.activ",
        "stage4.block1.expand.activ",
        "stage6.block1.expand.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
