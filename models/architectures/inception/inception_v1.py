"""
  # Description:
    - The following table comparing the params of the Inception v1 (GoogleNet) in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------------
      |        Model Name         |    Params       |
      |---------------------------------------------|
      |    Inception v1 naive     |   436,894,728   |
      |---------------------------------------------|
      |       Inception v1        |    56,146,392   |
       ---------------------------------------------

  # Reference:
    - [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
    - Source: https://github.com/guaiguaibao/GoogLeNet_Tensorflow2.0/tree/master/tensorflow2.0/GoogLeNet
              https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/GoogLeNet.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
    concatenate
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def inception_v1_naive_block(
    inputs,
    filters,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    """
    Inception module, naive version

    :param inputs: input tensor.
    :param blocks:
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_naive_block_{K.get_uid('inception_v1_naive_block')}"

    # branch1
    branch1 = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch1")(inputs)

    # branch2
    branch2 = Sequential([
        Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch2")(inputs)

    # branch3
    branch3 = Sequential([
        Conv2D(
            filters=filters[2],
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch3")(inputs)

    # branch4
    branch4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch4"
    )(inputs)

    merger = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return merger


def inception_v1_block(
    inputs,
    filters,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    """
    Inception module with dimension reductions

    :param inputs: input tensor.
    :param blocks: filter block, respectively: #1×1, #3×3 reduce, #3×3, #5×5 reduce, #5×5, pool proj
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_block_{K.get_uid('inception_v1_block')}"

    # branch 1x1
    branch_1x1 = Sequential([
        Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch1")(inputs)

    # branch 3x3
    branch_3x3 = Sequential([
        Conv2D(
            filters=filters[1],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[2],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch2")(inputs)

    # branch 5x5
    branch_5x5 = Sequential([
        Conv2D(
            filters=filters[3],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters[4],
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch3")(inputs)
    
    # branch pool
    
    branch_pool = Sequential([
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
        ),
        Conv2D(
            filters=filters[5],
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
    ], name=f"{name}.branch4")(inputs)

    merger = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis=-1, name=f"{name}.merger")
    return merger


def inception_v1_auxiliary_block(
    inputs,
    filters,
    num_classes,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.5,
    name=None
):
    """
    Inception auxiliary classifier module

    :param inputs: input tensor.
    :param num_classes: number off classes
    :return: Output tensor for the block.
    """
    if name is None:
        name = f"inception_v1_auxiliary_block_{K.get_uid('inception_v1_auxiliary_block')}"

    return Sequential([
        AveragePooling2D(pool_size=5, strides=3),
        Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Flatten(),
        Dropout(rate=drop_rate),
        Dense(units=1024),
        get_activation_from_name(activation),
        Dense(units=1 if num_classes == 2 else num_classes),
        get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
    ], name=name)(x)


def GoogleNet(
    block,
    auxiliary_logits=False,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
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
        min_size=64,
        weights=weights
    )

    # Stage 0
    x = Sequential([
        Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
    ], name="stem")(inputs)

    # Stage 1
    x = Sequential([
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
    ], name="stage1")(x)

    # Stage 2
    x = block(
        inputs=x,
        filters=[64, 96, 128, 16, 32, 32],
        **layer_constant_dict,
        name="stage2.block1"
    )
    
    x = block(
        inputs=x,
        filters=[128, 128, 192, 32, 96, 64],
        **layer_constant_dict,
        name="stage2.block2"
    )
    
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
        name="stage2.pool"
    )(x)

    # Stage 3
    x = block(
        inputs=x,
        filters=[192, 96, 208, 16, 48, 64],
        **layer_constant_dict,
        name="stage3.block1"
    )

    if auxiliary_logits:
        aux1 = inception_v1_auxiliary_block(x, classes)
        
    x = block(
        inputs=x,
        filters=[160, 112, 224, 24, 64, 64],
        **layer_constant_dict,
        name="stage3.block2"
    )
    
    x = block(
        inputs=x,
        filters=[128, 128, 256, 24, 64, 64],
        **layer_constant_dict,
        name="stage3.block3"
    )
    
    x = block(
        inputs=x,
        filters=[112, 144, 288, 32, 64, 64],
        **layer_constant_dict,
        name="stage3.block4"
    )
                  
    if auxiliary_logits:
        aux2 = inception_v1_auxiliary_block(x, classes)

    x = block(
        inputs=x,
        filters=[256, 160, 320, 32, 128, 128],
        **layer_constant_dict,
        name="stage3.block5"
    )
    
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same",
        name="stage3.pool"
    )(x)

    # Stage 4
    x = block(
        inputs=x,
        filters=[256, 160, 320, 32, 128, 128],
        **layer_constant_dict,
        name="stage4.block1"
    )
    
    x = block(
        inputs=x,
        filters=[384, 192, 384, 48, 128, 128],
        **layer_constant_dict,
        name="stage4.block2"
    )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same'),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    if auxiliary_logits:
        output = [aux1, aux2, x]
    else:
        output = x
        
    model = Model(inputs=inputs, outputs=output, name='Inception-v1')
    return model


def GoogleNet_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    model = GoogleNet(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def Inception_naive_v1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = GoogleNet(
        block=inception_v1_naive_block,
        auxiliary_logits=False,
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


def Inception_naive_v1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = Inception_naive_v1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def Inception_base_v1(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_rate=0.1
) -> Model:
    
    model = GoogleNet(
        block=inception_v1_block,
        auxiliary_logits=False,
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


def Inception_base_v1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = Inception_base_v1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage2.block2.merger",
        "stage3.block5.merger",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
