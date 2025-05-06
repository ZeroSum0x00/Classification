"""
  # Description:
    - The following table comparing the params of the SqueezeNet in Tensorflow on 
    size 224 x 224 x 3:

       -------------------------------------
      |     Model Name    |    Params       |
      |-------------------------------------|
      |    SqueezeNet     |    2,237,904    |
       -------------------------------------

  # Reference:
    - [SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/pdf/1602.07360.pdf)
    - Source: https://github.com/guaiguaibao/GoogLeNet_Tensorflow2.0/tree/master/tensorflow2.0/GoogLeNet
              https://github.com/marload/ConvNets-TensorFlow2/blob/master/models/SqueezeNet.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dropout, Dense, MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def fire_module(
    x,
    filters,
    squeeze_channel,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"fire_module_{K.get_uid('fire_module')}"

    x = Conv2D(
        filters=squeeze_channel,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv"
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"{name}.norm")(x)
    x = get_activation_from_name(activation, name=f"{name}.activ")(x)
    
    expand_1x1 = Sequential([
        Conv2D(
            filters=filters // 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation)
    ], name=f"{name}.expand_1x1")(x)
    
    expand_3x3 = Sequential([
        Conv2D(
            filters=filters // 2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
        ),
        get_normalizer_from_name(normalizer, epsilon=norm_eps),
        get_activation_from_name(activation)
    ], name=f"{name}.expand_3x3")(x)
    
    return concatenate([expand_1x1, expand_3x3], name=f"{name}.fusion")


def SqueezeNet(
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
    
    x = Conv2D(
        filters=96,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stem.conv"
    )(inputs)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="stem.norm")(x)
    x = get_activation_from_name(activation, name="stem.activ")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)

    for i in range(2):
        x = fire_module(
            x=x,
            filters=128,
            squeeze_channel=16,
            **layer_constant_dict,
            name=f"stage1.block{i+1}"
        )

    x = fire_module(
        x,
        filters=256,
        squeeze_channel=32,
        **layer_constant_dict,
        name="stage1.block3"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage1.pool")(x)

    x = fire_module(
        x=x,
        filters=256,
        squeeze_channel=32,
        **layer_constant_dict,
        name="stage2.block1"
    )
    
    for i in range(2):
        x = fire_module(
            x=x,
            filters=384,
            squeeze_channel=48,
            **layer_constant_dict,
            name=f"stage2.block{i+2}"
        )
        
    x = fire_module(
        x=x,
        filters=512,
        squeeze_channel=64,
        **layer_constant_dict,
        name="stage2.block4"
    )
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage2.pool")(x)

    x = fire_module(
        x=x,
        filters=512,
        squeeze_channel=64,
        **layer_constant_dict,
        name="stage3.block1"
    )
    
    x = Conv2D(
        filters=1 if num_classes == 2 else num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stage3.block2"
    )(x)
    
    x = AveragePooling2D((13, 13), strides=(1, 1), name="stage3.pool")(x)

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Flatten(name="flatten"),
            Dropout(drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    model = Model(inputs=inputs, outputs=x, name="SqueezeNet")
    return model


def SqueezeNet_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = SqueezeNet(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem.activ",
        "stage1.block3.fusion",
        "stage2.block4.fusion",
        "stage3.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
