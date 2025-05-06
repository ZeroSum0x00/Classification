"""
  # Description:
    - The following table comparing the params of the AlexNet in Tensorflow on 
    size 224 x 224 x 3:

       -------------------------------------
      |     Model Name    |    Params       |
      |-------------------------------------|
      |      AlexNet      |   50,844,008    |
       -------------------------------------

  # Reference:
    - [ImageNet Classification with Deep Convolutional Neural Networks
](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
    - Source: https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout, MaxPooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def AlexNet(
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
    drop_rate=0.5
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
        weights=weights,
    )

    x = Conv2D(
        filters=96,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stem.conv"
    )(inputs)
    
    x = get_activation_from_name(activation, name="stem.activ")(x)
    x = get_normalizer_from_name("local-response-norm", depth_radius=5, alpha=0.0001, beta=0.75, name="stem.norm")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stem.pool")(x)

    x = Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stage1.conv"
    )(x)
    
    x = get_activation_from_name(activation, name="stage1.activ")(x)
    x = get_normalizer_from_name("local-response-norm", depth_radius=5, alpha=0.0001, beta=0.75, name="stage1.norm")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage1.pool")(x)

    for i in range(2):
        x = Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name=f"stage2.conv{i+1}"
        )(x)
        
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name=f"stage2.norm{i+1}")(x)
        x = get_activation_from_name(activation, name=f"stage2.activ{i+1}")(x)

    
    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name="stage3.conv"
    )(x)
    
    x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="stage3.norm")(x)
    x = get_activation_from_name(activation, name="stage3.activ")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="stage3.pool")(x)

    if include_head:
        x = Sequential([
            Dropout(rate=drop_rate),
            Flatten(name="flatten"),
            Dense(units=4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            Dense(units=4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    model = Model(inputs=inputs, outputs=x, name="AlexNet")
    return model


def AlexNet_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="leaky-relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = AlexNet(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stem.norm",
        "stage1.norm",
        "stage3.activ",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
