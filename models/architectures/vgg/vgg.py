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


# BASE_WEIGTHS_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/')
# VGG16_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# VGG16_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
# VGG19_WEIGHT_PATH = (BASE_WEIGTHS_PATH + 'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
# VGG19_WEIGHT_PATH_NO_TOP = (BASE_WEIGTHS_PATH + 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGGBlock(x,
             filters,
             num_blocks,
             activation='relu',
             normalizer='batch-norm',
             kernel_initializer="he_normal",
             bias_initializer="zeros",
             regularizer_decay=5e-4,
             norm_eps=1e-6,
             name='vgg_block'):
    for i in range(num_blocks):
        x = Sequential([
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l2(regularizer_decay),
            ),
            get_normalizer_from_name(normalizer, epsilon=norm_eps),
            get_activation_from_name(activation),
        ], name=name + f".block{i + 1}")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name + ".pool")(x)
    return x


def VGG(
    filters,
    num_blocks,
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
    x = VGGBlock(
        x=inputs,
        filters=filters,
        num_blocks=num_blocks[0],
        **layer_constant_dict,
        name='stem'
    )

    # Stage 1:
    x = VGGBlock(
        x=x,
        filters=filters * 2,
        num_blocks=num_blocks[1],
        **layer_constant_dict,
        name='stage1'
    )

    # Stage 2:
    x = VGGBlock(
        x=x,
        filters=filters * 4,
        num_blocks=num_blocks[2],
        **layer_constant_dict,
        name='stage2'
    )

    # Stage 3:
    x = VGGBlock(
        x=x,
        filters=filters * 8,
        num_blocks=num_blocks[3],
        **layer_constant_dict,
        name='stage3'
    )

    # Stage 4:
    x = VGGBlock(
        x=x,
        filters=filters * 8,
        num_blocks=num_blocks[4],
        **layer_constant_dict,
        name='stage4'
    )

    if include_head:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc1')(x)
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps)(x)
        x = get_activation_from_name(activation)(x)
        x = Dropout(rate=drop_rate)(x)
        
        x = Dense(4096, name='fc2')(x)
        x = get_normalizer_from_name(normalizer, epsilon=norm_eps)(x)
        x = get_activation_from_name(activation)(x)
        x = Dropout(rate=drop_rate)(x)

        x = Dense(1 if num_classes == 2 else num_classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    # Create model.
    if filters == 64 and num_blocks == [1, 1, 2, 2, 2]:
        model = Model(inputs=inputs, outputs=x, name='VGG-11')
        model_type = 1
    elif filters == 64 and num_blocks == [2, 2, 2, 2, 2]:
        model = Model(inputs=inputs, outputs=x, name='VGG-13')
        model_type = 2
    elif filters == 64 and num_blocks == [2, 2, 3, 3, 3]:
        model = Model(inputs=inputs, outputs=x, name='VGG-16')
        model_type = 3
    elif filters == 64 and num_blocks == [2, 2, 4, 4, 4]:
        model = Model(inputs=inputs, outputs=x, name='VGG-19')
        model_type = 4
    else:
        model = Model(inputs=inputs, outputs=x, name='VGG')
        model_type = 0

    # Load weights.
    if weights == 'imagenet':
        # if include_head:
        #     if model_type == 1:
        #         weights_path = None
        #     elif model_type == 2:
        #         weights_path = None
        #     elif model_type == 3:
        #         weights_path = get_file(
        #             'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        #             VGG16_WEIGHT_PATH,
        #             cache_subdir='models',
        #             file_hash='64373286793e3c8b2b4e3219cbf3544b')
        #     elif model_type == 4:
        #         weights_path = get_file(
        #             'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
        #             VGG19_WEIGHT_PATH,
        #             cache_subdir='models',
        #             file_hash='cbe5617147190e668d6c5d5026f83318')
        # else:
        #     if model_type == 1:
        #         weights_path = None
        #     elif model_type == 2:
        #         weights_path = None
        #     elif model_type == 3:
        #         weights_path = get_file(
        #             'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #             VGG16_WEIGHT_PATH_NO_TOP,
        #             cache_subdir='models',
        #             file_hash='6d6bbae143d832006294945121d1f1fc')
        #     elif model_type == 4:
        #         weights_path = get_file(
        #             'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #             VGG19_WEIGHT_PATH_NO_TOP,
        #             cache_subdir='models',
        #             file_hash='253f8cb515780f3b799900260a226db6')
        # if weights_path is not None:
        #     model.load_weights(weights_path)
        pass
    elif weights is not None:
        model.load_weights(weights)

    return model


def VGG_backbone(
    filters,
    num_blocks,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
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
    normalizer="batch-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    deploy=False,
    regularizer_decay=5e-4,
    norm_eps=1e-6,
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
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def VGG11_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = VGG11(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
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
    normalizer="batch-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    deploy=False,
    regularizer_decay=5e-4,
    norm_eps=1e-6,
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
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def VGG13_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = VGG13(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
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
    normalizer="batch-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    deploy=False,
    regularizer_decay=5e-4,
    norm_eps=1e-6,
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
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def VGG16_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = VGG16(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
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
    normalizer="batch-norm",
    final_activation="softmax",
    num_classes=1000,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    deploy=False,
    regularizer_decay=5e-4,
    norm_eps=1e-6,
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
        final_activation=final_activation,
        num_classes=num_classes,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        drop_rate=drop_rate,
    )
    return model


def VGG19_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:
    
    model = VGG19(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer
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
    