"""
  # Description:
    - The following table comparing the params of the DenseNet in Tensorflow on 
    size 224 x 224 x 3:

       --------------------------------------
      |      Model Name     |    Params      |
      |--------------------------------------|
      |     DenseNet-121    |    8,062,504   |
      |---------------------|----------------|
      |     DenseNet-169    |   14,307,880   |
      |---------------------|----------------|
      |     DenseNet-201    |   20,242,984   |
      |---------------------|----------------|
      |     DenseNet-264    |   33,736,232   |
       --------------------------------------

  # Reference:
    - [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
              https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, Dense, MaxPooling2D, Dropout,
    AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
    concatenate
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input


def conv_block(
    inputs,
    growth_rate,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A building block for a dense block

    :param inputs: input tensor.
    :param growth_rate: float, growth rate at dense layers.
    :param name: string, block label.
    :return: Output tensor for the block.
    """    
    if name is None:
        name = f"conv_block_{K.get_uid('conv_block')}"

    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm1")(inputs)
    x = get_activation_from_name(activation, name=f"{name}.activ1")(x)
    
    x = Conv2D(
        filters=growth_rate * 4,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv1",
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm2")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ2")(x)
    
    x = Conv2D(
        filters=growth_rate,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv2",
    )(x)
    
    merge = concatenate([inputs, x], name=f"{name}.merger")
    return merge


def dense_block(
    inputs,
    blocks,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A dense block.

    :param inputs: input tensor.
    :param blocks: integer, the number of building blocks.
    :param name: string, block label.
    :return: output tensor for the block.
    """
    if name is None:
        name = f"dense_block_{K.get_uid('dense_block')}"

    x = inputs
    for i in range(blocks):
        x = conv_block(
            inputs=x,
            growth_rate=32,
            activation=activation,
            normalizer=normalizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            regularizer_decay=regularizer_decay,
            norm_eps=norm_eps,
            name=f"{name}.step{str(i + 1)}",
        )
    return x


def transition_block(
    inputs,
    reduction,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None,
):
    """
    A transition block.

    :param inputs: input tensor.
    :param reduction: float, compression rate at transition layers.
    :param name: string, block label.
    :return: output tensor for the block.
    """    
    if name is None:
        name = f"transition_block_{K.get_uid('transition_block')}"

    channel_axis = -1
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(inputs)
    x = get_activation_from_name(activation, name=f"{name}.activ")(x)
    
    x = Conv2D(
        filters=int(K.int_shape(x)[channel_axis] * reduction),
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv",
    )(x)
    
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=f"{name}.pool")(x)
    return x


def DenseNet(
    blocks,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer="batch-norm",
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
        weights=weights
    )


    # Stage 0:
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name="stem.padding1")(inputs)
    
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stem.conv",
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="stem.norm")(x)
    x = get_activation_from_name(activation, name="stem.activ")(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name="stem.padding2")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)


    for i, block in enumerate(blocks):
        x = dense_block(
            inputs=x,
            blocks=block,
            **layer_constant_dict,
            name=f"stage{i + 1}.block1"
        )

        if i != len(blocks) - 1:
            x = transition_block(
                inputs=x,
                reduction=0.5,
                **layer_constant_dict,
                name=f"stage{i + 1}.block2"
            )
        else:
            x = Sequential([
                get_normalizer_from_name(normalizer, epsilon=norm_eps),
                get_activation_from_name(activation)
            ], name=f"stage{i + 1}.block2")(x)
                 
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

    if blocks == [6, 12, 24, 16]:
        model = Model(inputs=inputs, outputs=x, name="DenseNet-121")
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs=inputs, outputs=x, name="DenseNet-169")
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs=inputs, outputs=x, name="DenseNet-201")
    elif blocks == [6, 12, 64, 48]:
        model = Model(inputs=inputs, outputs=x, name="DenseNet-264")
    else:
        model = Model(inputs=inputs, outputs=x, name="DenseNet")

    return model


def DenseNet_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = DenseNet(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block2.conv",
        "stage2.block2.conv",
        "stage3.block2.conv",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DenseNet121(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = DenseNet(
        blocks=[6, 12, 24, 16],
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
        drop_rate=drop_rate
    )
    return model


def DenseNet121_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = DenseNet121(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block2.conv",
        "stage2.block2.conv",
        "stage3.block2.conv",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DenseNet169(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:

    model = DenseNet(
        blocks=[6, 12, 32, 32],
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
        drop_rate=drop_rate
    )
    return model


def DenseNet169_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = DenseNet169(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block2.conv",
        "stage2.block2.conv",
        "stage3.block2.conv",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DenseNet201(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:

    model = DenseNet(
        blocks=[6, 12, 48, 32],
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
        drop_rate=drop_rate
    )
    return model


def DenseNet201_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = DenseNet201(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block2.conv",
        "stage2.block2.conv",
        "stage3.block2.conv",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def DenseNet264(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="leaky-relu",
    normalizer="batch-norm",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:

    model = DenseNet(
        blocks=[6, 12, 64, 48],
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
        drop_rate=drop_rate
    )
    return model


def DenseNet264_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="silu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = DenseNet264(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block2.conv",
        "stage2.block2.conv",
        "stage3.block2.conv",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
