"""
  # Description:
    - The following table comparing the params of the VGGNet in Tensorflow on 
    size 224 x 224 x 3:

       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |       VGG-11        |   132,863,336   |
      |---------------------|-----------------|
      |       VGG-13        |   133,047,848   |
      |---------------------|-----------------|
      |       VGG-16        |   138,357,544   |
      |---------------------|-----------------|
      |       VGG-19        |   143,667,240   |
       ---------------------------------------

  # Reference:
    - [Very deep convolutional networks for large-scale image 
       recognition](https://arxiv.org/pdf/1409.1556.pdf)
    - Source: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
              https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Dropout, MaxPooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input



def VGGBlock(
    inputs,
    filters,
    num_blocks,
    activation="relu",
    normalizer=None,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
    name=None,
):
    if name is None:
        name = f"vgg_block_{K.get_uid('vgg_block')}"

    if regularizer_decay and regularizer_decay > 0:
        kernel_regularizer = l2(regularizer_decay)
    else:
        kernel_regularizer = None
        
    x = inputs
    for i in range(num_blocks):
        x = Sequential([
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
        ], name=f"{name}.block{i + 1}")(x)
        
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"{name}.pool")(x)
    return x


def VGG(
    filters,
    num_blocks,
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
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
    
    filters = filters if isinstance(filters, (tuple, list)) else [filters * 2**i for i in range(len(num_blocks) - 1)]

    inputs = process_model_input(
        inputs,
        include_head=include_head,
        default_size=224,
        min_size=32,
        weights=weights,
    )
    
    x = VGGBlock(
        inputs=inputs,
        filters=filters[0],
        num_blocks=num_blocks[0],
        **layer_constant_dict,
        name="stem"
    )
    
    for i in range(len(num_blocks) - 1):
        x = VGGBlock(
            inputs=x,
            filters=filters[i + 1] if i <= len(num_blocks) - 3 else filters[i],
            num_blocks=num_blocks[i + 1],
            **layer_constant_dict,
            name=f"stage{i + 1}",
        )

    if include_head:
        x = Sequential([
            Flatten(),
            Dense(4096),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
            Dropout(rate=drop_rate),
            Dense(4096),
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

    if filters == [64, 128, 256, 512]:
        if num_blocks == [1, 1, 2, 2, 2]:
            model = Model(inputs=inputs, outputs=x, name="VGG-11")
        elif num_blocks == [2, 2, 2, 2, 2]:
            model = Model(inputs=inputs, outputs=x, name="VGG-13")
        elif num_blocks == [2, 2, 3, 3, 3]:
            model = Model(inputs=inputs, outputs=x, name="VGG-16")
        elif num_blocks == [2, 2, 4, 4, 4]:
            model = Model(inputs=inputs, outputs=x, name="VGG-19")
        else:
            model = Model(inputs=inputs, outputs=x, name="VGG")
    else:
        model = Model(inputs=inputs, outputs=x, name="VGG")
        
    return model


def VGG_backbone(
    filters,
    num_blocks,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:

    model = VGG(
        filters=filters,
        num_blocks=num_blocks,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
    )
    
    custom_layers = custom_layers or [
        "stage1.pool",
        "stage2.pool",
        "stage3.pool",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def VGG11(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
    drop_rate=0.1
) -> Model:
    
    model = VGG(
        filters=64,
        num_blocks=[1, 1, 2, 2, 2],
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


def VGG11_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = VGG11(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block1",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def VGG13(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
    drop_rate=0.1
) -> Model:
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 2, 2, 2],
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


def VGG13_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = VGG13(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block2",
        "stage3.block2",
        "stage4.block2",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def VGG16(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
    drop_rate=0.1
) -> Model:
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 3, 3, 3],
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


def VGG16_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = VGG16(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block3",
        "stage3.block3",
        "stage4.block3",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def VGG19(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="relu",
    normalizer=None,
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=0,
    norm_eps=0.001,
    drop_rate=0.1
) -> Model:
    
    model = VGG(
        filters=64,
        num_blocks=[2, 2, 4, 4, 4],
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


def VGG19_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer=None,
    custom_layers=[]
) -> Model:
    
    model = VGG19(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stage1.block2",
        "stage2.block4",
        "stage3.block4",
        "stage4.block4",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
    