"""
  # Description:
    - The following table comparing the params of the Inception Resnet v1 in Tensorflow on 
    size 299 x 299 x 3:

       ---------------------------------------------
      |         Model Name        |     Params      |
      |---------------------------------------------|
      |    Inception Resnet v1    |   136,038,104   |
       ---------------------------------------------

  # Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
    - Source: https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v2.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Lambda, Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
    concatenate, add
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def convolution_block(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    activation="relu",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    name=None
):
    if name is None:
        name = f"convolution_block_{K.get_uid('convolution_block')}"

    return Sequential([
        Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_activation_from_name(activation),
    ])(inputs)


def stem_block(
    inputs,
    filters=32,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"stem_block_{K.get_uid('stem_block')}"

    x = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv1"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv2"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv3"
    )
    
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.pool"
    )(x)
    
    x = convolution_block(
        inputs=x,
        filters=int(filters * 5/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv4"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv5"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 8,
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv6"
    )
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ")(x)
    return x


def inception_resnet_A(
    inputs,
    filters=32,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_a_{K.get_uid('inception_resnet_a')}"
        
    shortcut = inputs

    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step3"
    )
    
    # merger
    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 8,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)
        
    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def reduction_A(
    inputs,
    k=192,
    l=192,
    m=256,
    n=384,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"reduction_a_{K.get_uid('reduction_a')}"

    # branch1
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=n,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2"
    )

    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=k,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=l,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=m,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step3"
    )

    # merger
    out = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def inception_resnet_B(
    inputs,
    filters=128,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_b_{K.get_uid('inception_resnet_b')}"

    shortcut = inputs

    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )
    
    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step3"
    )

    # merger
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 7,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)

    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def reduction_B(
    inputs,
    filters=256,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"reduction_b_{K.get_uid('reduction_b')}"

    # branch1
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)
    
    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    # branch3
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch3.step2"
    )

    # branch4
    branch4 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch4.step3"
    )

    # merger
    out = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def inception_resnet_C(
    inputs,
    filters=128,
    scale_residual=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_resnet_c_{K.get_uid('inception_resnet_c')}"
        
    shortcut = inputs
    
    # branch1
    branch1 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch1"
    )

    # branch2
    branch2 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=int(filters * 3/2),
        kernel_size=(1, 3),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=int(filters * 3/2),
        kernel_size=(3, 1),
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.branch2.step3"
    )

    # merger
    x = concatenate([branch1, branch2], axis=-1, name=f"{name}.merger")
    
    x = convolution_block(
        inputs=x,
        filters=filters * 14,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        name=f"{name}.conv"
    )
    
    
    if scale_residual:
        x = Lambda(lambda s: s * 0.1, name=f"{name}.scale")(x)
        
    out = add([shortcut, x], name=f"{name}.residual")
    out = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(out)
    out = get_activation_from_name(activation, name=f"{name}.activ")(out)
    return out


def Inception_Resnet_v1(
    filters,
    num_blocks,
    scale_residual,
    inputs=[299, 299, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer="batch-norm",
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
        default_size=299,
        min_size=64,
        weights=weights
    )

    # Stem block
    x = stem_block(
        inputs=inputs,
        filters=filters,
        **layer_constant_dict,
        name="stem"
    )

    # Inception-A
    for i in range(num_blocks[0]):
        x = inception_resnet_A(
            inputs=x,
            filters=filters,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    # Reduction-A
    x = reduction_A(
        inputs=x,
        k=192,
        l=192,
        m=256,
        n=384,
        **layer_constant_dict,
        name=f"stage1.block{i + 2}"
    )

    # Inception-B
    for i in range(num_blocks[1]):
        x = inception_resnet_B(
            inputs=x,
            filters=filters * 4,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    # Reduction-B
    x = reduction_B(
        inputs=x,
        filters=filters * 8,
        **layer_constant_dict,
        name=f"stage2.block{i + 2}"
    )
                  
    # Inception-C
    for i in range(num_blocks[2]):
        x = inception_resnet_C(
            inputs=x,
            filters=filters * 4,
            scale_residual=scale_residual,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same"),
            Dropout(rate=drop_rate),
            Flatten(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="global_avg_pooling")(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D(name="global_max_pooling")(x)

    model = Model(inputs=inputs, outputs=x, name="Inception-Resnet-v1")
    return model


def Inception_Resnet_v1_backbone(
    filters,
    num_blocks,
    scale_residual,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = Inception_Resnet_v1(
        filters=filters,
        num_blocks=num_blocks,
        scale_residual=scale_residual,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.conv3.activ",
        "stem.conv5.activ",
        f"stage1.block{num_blocks[0] + 1}.branch3.step2.activ",
        f"stage2.block{num_blocks[1] + 1}.branch4.step2.activ",
    ]
    
    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def Inception_Resnet_base_v1(
    inputs=[299, 299, 3],
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
) -> Model:
    
    model = Inception_Resnet_v1(
        filters=32,
        num_blocks=[5, 10, 5],
        scale_residual=True,
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


def Inception_Resnet_base_v1_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = Inception_Resnet_base_v1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.conv3.activ",
        "stem.conv5.activ",
        "stage1.block6.branch3.step2.activ",
        "stage2.block11.branch4.step2.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
