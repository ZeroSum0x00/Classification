"""
  # Description:
    - On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
    and a top-5 validation accuracy of 0.945.
    - Also do note that this model is only available for the TensorFlow backend,
    due to its reliance on `SeparableConvolution` layers.
    - The following table comparing the params of the Extreme Inception (Xception) in 
    Tensorflow on size 299 x 299 x 3:

       --------------------------------------
      |     Model Name      |    Params      |
      |--------------------------------------|
      |     Xception        |   23,238,312   |
       --------------------------------------

  # Reference:
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, Dense, Dropout,
    MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D,
    add
)
from tensorflow.keras.regularizers import l2

from .inception_v3 import convolution_block
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def xception_block(
    inputs,
    filters,
    shortcut=True,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"xception_block_{K.get_uid('xception_block')}"

    x = Sequential([
        get_activation_from_name(activation),
        SeparableConv2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=l2(regularizer_decay),
            pointwise_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        SeparableConv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=l2(regularizer_decay),
            pointwise_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name=f"{name}.separable")(inputs)

    if shortcut:
        residual = Sequential([
            Conv2D(
                filters=filters[1],
                kernel_size=(1, 1),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l2(regularizer_decay),
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"{name}.residual")(inputs)
    
        x = MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
            name=f"{name}.pooling"
        )(x)
    
        x = add([x, residual], name=f"{name}.add")
        
    return x


def Xception(
    filters,
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
        min_size=71,
        weights=weights
    )
    
    # Block 1
    x = Sequential([
        Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation),
        Conv2D(
            filters=filters * 2,
            kernel_size=(3, 3),
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
    ], name="stem")(inputs)

    x = xception_block(x, filters=[filters * 4, filters * 4], name="stage1")
    x = xception_block(x, filters=[filters * 8, filters * 8], name="stage2")
    x = xception_block(x, filters=[filters * 23, filters * 23], name="stage3")

    for i in range(8):
        residual = x
        x = Sequential([
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=l2(regularizer_decay),
                pointwise_regularizer=l2(regularizer_decay),
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=l2(regularizer_decay),
                pointwise_regularizer=l2(regularizer_decay),
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            SeparableConv2D(
                filters=filters * 23,
                kernel_size=(3, 3),
                padding='same',
                use_bias=False,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=l2(regularizer_decay),
                pointwise_regularizer=l2(regularizer_decay),
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
        ], name=f"stage4.block{i + 1}.separable")(x)
        
        x = add([x, residual], name=f"stage4.block{i + 1}.add")

    x = xception_block(x, filters=[filters * 23, filters * 32], name="stage5")
    x = xception_block(x, filters=[filters * 48, filters * 64], shortcut=False, name="stage6")

    if include_head:
        x = Sequential([
            GlobalAveragePooling2D(),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avgpool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_maxpool')(x)

    model = Model(inputs, x, name='Xception')
    return model


def Xception_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = Xception(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def Xception_base(
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
    
    model = Xception(
        filters=32,
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

def Xception_base_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = Xception_base(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem",
        "stage1.add",
        "stage2.add",
        "stage4.block8.add",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
