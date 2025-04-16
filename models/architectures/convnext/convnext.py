"""
  # Description:
    - The following table comparing the params of the ConvNeXt in Pytorch Source 
    with Tensorflow convert Source on size 224 x 224 x 3:
      
       ---------------------------------------
      |     Model Name      |    Params       |
      |---------------------------------------|
      |     ConvNeXt-T      |    28,582,504   |
      |---------------------------------------|
      |     ConvNeXt-S      |    50,210,152   |
      |---------------------------------------|
      |     ConvNeXt-B      |    88,573,416   |
      |---------------------------------------|
      |     ConvNeXt-L      |   197,740,264   |
      |---------------------------------------|
      |     ConvNeXt-XL     |   350,160,872   |
       ---------------------------------------

  # Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf)
    - Source: https://github.com/facebookresearch/ConvNeXt

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input, create_layer_instance

try:
    from models.layers import StochasticDepth
except ImportError:
    from models.layers import StochasticDepth2 as StochasticDepth



def stem_cell(
    inputs,
    out_filter,
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"stem_cell{K.get_uid('stem_cell')}"
    
    x = Conv2D(
        filters=out_filter, 
        kernel_size=(4, 4), 
        strides=(4, 4), 
        padding="same", 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv"
    )(inputs)
    
    x = get_normalizer_from_name(
        normalizer,
        epsilon=norm_eps,
        name=f"{name}.norm"
    )(x)
    
    return x


def Downsample(
    inputs, 
    out_filter, 
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"downsample_block{K.get_uid('downsample')}"
        
    x = get_normalizer_from_name(
        normalizer,
        epsilon=norm_eps,
        name=f"{name}.norm"
    )(inputs)
    
    x = Conv2D(
        filters=out_filter,
        kernel_size=(2, 2), 
        strides=(2, 2), 
        padding="same",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv"
    )(x)
    
    return x


def ConvNextBlock(
    inputs, 
    drop_prob=0, 
    layer_scale_init_value=1e-6, 
    activation="gelu",
    normalizer="layer-norm",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"conv_next_block{K.get_uid('conv_next_block')}"
  
    in_filters = inputs.shape[-1]

    x = Conv2D(
        filters=in_filters,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding="same",
        groups=in_filters,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.dw"
    )(inputs)
    
    x = get_normalizer_from_name(
        normalizer,
        epsilon=norm_eps,
        name=f"{name}.norm"
    )(x)
    
    x = Conv2D(
        filters=in_filters*4,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.pw"
    )(x)
    
    x = get_activation_from_name(activation, name=f"{name}.activ")(x)
    
    x = Conv2D(
        filters=in_filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=l2(regularizer_decay),
        name=f"{name}.conv"
    )(x)

    if layer_scale_init_value > 0:
        layer_scale_gamma = tf.ones(in_filters) * layer_scale_init_value
        x = x * layer_scale_gamma

    if drop_prob > 0:
        x = StochasticDepth(drop_prob, name=f"{name}.drop_path")([inputs, x])
    
    x = Lambda(lambda x: x, name=f"{name}.final")(x)
    return x


def ConvNext(
    filters,
    num_blocks,
    layer_scale_init_value,
    inputs=[224, 224, 3],
    include_head=True, 
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
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

    cur = 0
    dp_rates = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(num_blocks))]

    x = create_layer_instance(
        stem_cell,
        inputs=inputs,
        out_filter=filters,
        **layer_constant_dict,
        name="stem"
    )
    
    for i in range(num_blocks[0]):
        x = create_layer_instance(
            ConvNextBlock,
            inputs=x, 
            drop_prob=dp_rates[cur + i],
            layer_scale_init_value=layer_scale_init_value,
            **layer_constant_dict,
            name=f"stage1.block{str(i+1)}"
        )
    
    cur += num_blocks[0]
    x = create_layer_instance(
        Downsample,
        inputs=x,
        out_filter=filters * 2,
        **layer_constant_dict,
        name="stage2.block1"
    )
    
    for i in range(num_blocks[1]):
        x = create_layer_instance(
            ConvNextBlock,
            inputs=x, 
            drop_prob=dp_rates[cur + i], 
            layer_scale_init_value=layer_scale_init_value,
            **layer_constant_dict,
            name=f"stage2.block{str(i+2)}"
        )

    cur += num_blocks[1]
    x = create_layer_instance(
        Downsample,
        inputs=x,
        out_filter=filters * 4,
        **layer_constant_dict,
        name="stage3.block1"
    )
    
    for i in range(num_blocks[2]):
        x = create_layer_instance(
            ConvNextBlock,
            inputs=x, 
            drop_prob=dp_rates[cur + i], 
            layer_scale_init_value=layer_scale_init_value,
            **layer_constant_dict,
            name=f"stage3.block{str(i+2)}"
        )

    cur += num_blocks[2]
    x = create_layer_instance(
        Downsample,
        inputs=x,
        out_filter=filters * 8,
        **layer_constant_dict,
        name="stage4.block1"
    )
    
    for i in range(num_blocks[3]):
        x = create_layer_instance(
            ConvNextBlock,
            inputs=x, 
            drop_prob=dp_rates[cur + i], 
            layer_scale_init_value=layer_scale_init_value,
            **layer_constant_dict,
            name=f"stage4.block{str(i+2)}"
        )

    if include_head:
        x = GlobalAveragePooling2D(name="global_avgpool")(x)
        x = Dropout(rate=drop_rate)(x)
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="final_norm")(x)
        x = Dense(
            units=1 if num_classes == 2 else num_classes, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer,
            kernel_regularizer=l2(regularizer_decay),
            name="predictions"
        )(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D(name="global_avgpool")(x)
            x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="final_norm")(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D(name="global_maxpool")(x)
            x = get_normalizer_from_name(normalizer, epsilon=norm_eps, name="final_norm")(x)

    if filters == 96 and num_blocks == [3, 3, 9, 3]:
        model = Model(inputs, x, name="ConvNext-T")
    elif num_blocks == [3, 3, 27, 3]:
        if filters ==96:
            model = Model(inputs, x, name="ConvNext-S")
        elif filters ==128:
            model = Model(inputs, x, name="ConvNext-B")
        elif filters ==192:
            model = Model(inputs, x, name="ConvNext-L")
        elif filters == 256:
            model = Model(inputs, x, name="ConvNext-XL")
        else:
            model = Model(inputs, x, name="ConvNext")
    else:
        model = Model(inputs, x, name="ConvNext")

    return model


def ConvNext_backbone(
    filters,
    num_blocks,
    layer_scale_init_value,
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNext(
        filters=filters,
        num_blocks=num_blocks,
        layer_scale_init_value=layer_scale_init_value,
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        f"stage1.block{num_blocks[0] + 1}.final",
        f"stage2.block{num_blocks[1] + 2}.final",
        f"stage3.block{num_blocks[2] + 2}.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def ConvNextT(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
    drop_rate=0.1
) -> Model:
    
    model = ConvNext(
        filters=96,
        num_blocks=[3, 3, 9, 3],
        layer_scale_init_value=1e-6,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNextT_backbone(
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNextT(
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block10.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def ConvNextS(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
    drop_rate=0.1
) -> Model:
    
    model = ConvNext(
        filters=96,
        num_blocks=[3, 3, 27, 3],
        layer_scale_init_value=1e-6,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNextS_backbone(
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNextS(
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def ConvNextB(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
    drop_rate=0.1
) -> Model:
    
    model = ConvNext(
        filters=128,
        num_blocks=[3, 3, 27, 3],
        layer_scale_init_value=1e-6,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNextB_backbone(
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNextB(
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def ConvNextL(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
    drop_rate=0.1
) -> Model:
    
    model = ConvNext(
        filters=192,
        num_blocks=[3, 3, 27, 3],
        layer_scale_init_value=1e-6,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNextL_backbone(
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNextL(
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def ConvNextXL(
    inputs=[224, 224, 3],
    include_head=True,
    weights="imagenet",
    pooling=None,
    activation="gelu",
    normalizer="layer-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    drop_path_rate=0.,
    drop_rate=0.1
) -> Model:
    
    model = ConvNext(
        filters=256,
        num_blocks=[3, 3, 27, 3],
        layer_scale_init_value=1e-6,
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
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate
    )
    return model


def ConvNextXL_backbone(
    inputs=[224, 224, 3],
    weights="imagenet", 
    pooling=None, 
    activation="gelu",
    normalizer="layer-norm",
    custom_layers=[],
) -> Model:

    model = ConvNextXL(
        inputs=inputs,
        include_head=False,
        weights=weights,
        pooling=pooling,
        activation=activation,
        normalizer=normalizer
    )

    custom_layers = custom_layers or [
        "stage1.block3.final",
        "stage2.block4.final",
        "stage3.block28.final",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


if __name__ == "__main__":
    model = ConvNextL(input_shape=(224, 224, 3), weights=None)
    model.summary()