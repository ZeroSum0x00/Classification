"""
  # Description:
    - The following table comparing the params of the Inception v4 in Tensorflow on 
    size 299 x 299 x 3:

       --------------------------------------------
      |        Model Name        |    Params       |
      |--------------------------------------------|
      |       Inception v4       |   139,456,552   |
       --------------------------------------------

  # Reference:
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
    - Source: https://github.com/titu1994/Inception-v4/blob/master/inception_v4.py

"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Flatten, Dense, Dropout,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D, GlobalAveragePooling2D,
    concatenate
)
from tensorflow.keras.regularizers import l2

from .inception_v3 import convolution_block
from models.layers import get_activation_from_name, get_normalizer_from_name
from utils.model_processing import process_model_input


def stem_block(
    inputs,
    filters,
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
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block1"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block2"
    )
    
    x = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block3"
    )

    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stem.block4.branch1.pooling"
    )(x)
    
    branch2 = convolution_block(
        inputs=x,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block4.branch2.conv"
    )
    
    x = concatenate([branch1, branch2], axis=-1, name='stem.merger1')
    
    branch1 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch1.conv1"
    )
    
    branch1 = convolution_block(
        branch1,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch1.conv2"
    )

    branch2 = convolution_block(
        inputs=x,
        filters=filters * 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv2"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 2,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv3"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=filters * 3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block5.branch2.conv4"
    )

    x = concatenate([branch1, branch2], axis=-1, name='stem.merger2')

    branch1 = convolution_block(
        inputs=x,
        filters=filters * 6,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name="stem.block6.branch1.conv"
    )
    
    branch2 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="stem.block6.branch2.pooling"
    )(x)
    
    x = concatenate([branch1, branch2], axis=-1, name='stem.merger3')
    return x


def inception_A(
    inputs,
    filters,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_block_a_{K.get_uid('inception_block_a')}"

    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=int(filters * 3/2),
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )

    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )

    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=int(filters * 3/2),
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f'{name}.merger')
    return x


def reduction_A(
    inputs,
    k=192,
    l=224,
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

    # Branch 1:
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        name=f"{name}.branch1"
    )(inputs)
    
    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=n,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=k,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=l,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=m,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.step3"
    )

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return x


def inception_B(
    inputs,
    filters,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_b_{K.get_uid('inception_b')}"

    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=128,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=384,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=192,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=224,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=256,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=192,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=192,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=224,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=224,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv4"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=256,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv5"
    )
    
    x = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}.merger")
    return x


def reduction_B(
    inputs,
    filters,
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
        
    # Branch 1:
    branch1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name=f"{name}.branch1.pooling"
    )(inputs)

    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=192,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv1"
    )
    
    branch2 = convolution_block(
        inputs=branch2,
        filters=192,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv2"
    )
    
    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=256,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=256,
        kernel_size=(1, 7),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=320,
        kernel_size=(7, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    branch3 = convolution_block(
        inputs=branch3,
        filters=320,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv4"
    )

    x = concatenate([branch1, branch2, branch3], axis=-1, name=f"{name}.merger")
    return x


def inception_C(
    inputs,
    filters,
    activation="relu",
    normalizer="batch-norm",
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    regularizer_decay=5e-4,
    norm_eps=1e-6,
    name=None
):
    if name is None:
        name = f"inception_c_{K.get_uid('inception_c')}"
        
    # Branch 1:
    branch1 = AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
        name=f"{name}.branch1.pooling"
    )(inputs)
    
    branch1 = convolution_block(
        inputs=branch1,
        filters=256,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch1.conv"
    )
    
    # Branch 2:
    branch2 = convolution_block(
        inputs=inputs,
        filters=256,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch2.conv"
    )
    

    # Branch 3:
    branch3 = convolution_block(
        inputs=inputs,
        filters=384,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv1"
    )
    
    branch31 = convolution_block(
        inputs=branch3,
        filters=256,
        kernel_size=(1, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv2"
    )
    
    branch32 = convolution_block(
        inputs=branch3,
        filters=256,
        kernel_size=(3, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch3.conv3"
    )
    
    # Branch 4:
    branch4 = convolution_block(
        inputs=inputs,
        filters=384,
        kernel_size=(1, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv1"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=448,
        kernel_size=(1, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv2"
    )
    
    branch4 = convolution_block(
        inputs=branch4,
        filters=512,
        kernel_size=(3, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv3"
    )
    
    branch41 = convolution_block(
        inputs=branch4,
        filters=256,
        kernel_size=(1, 3),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv4"
    )
    
    branch42 = convolution_block(
        inputs=branch4,
        filters=256,
        kernel_size=(3, 1),
        strides=(1, 1),
        activation=activation, 
        normalizer=normalizer, 
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        regularizer_decay=regularizer_decay,
        norm_eps=norm_eps,
        name=f"{name}.branch4.conv5"
    )
    
    x = concatenate([branch1, branch2, branch31, branch32, branch41, branch42], axis=-1, name=f'{name}.merger')
    return x


def Inception_v4(
    num_blocks,
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
        filters=32,
        **layer_constant_dict,
        name="stem"
    )
    
    # Inception-A
    for i in range(num_blocks[0]):
        x = inception_A(
            inputs=x,
            filters=64,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )

    x = reduction_A(
        inputs=x,
        k=192,
        l=224,
        m=256,
        n=384,
        **layer_constant_dict,
        name=f"stage1.block{i + 2}"
    )

    # Inception-B
    for i in range(num_blocks[1]):
        x = inception_B(
            inputs=x,
            filters=128,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )

    # Reduction-B
    x = reduction_B(
        inputs=x,
        filters=192,
        **layer_constant_dict,
        name=f"stage2.block{i + 2}"
    )
                  
    # Inception-C
    for i in range(num_blocks[2]):
        x = inception_C(
            inputs=x,
            filters=256,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )

    if include_head:
        x = Sequential([
            AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='same'),
            Dropout(rate=drop_rate),
            Flatten(),
            Dense(units=1 if num_classes == 2 else num_classes),
            get_activation_from_name("sigmoid" if num_classes == 2 else "softmax"),
        ], name="classifier_head")(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_max_pooling')(x)
            
    if num_blocks == [4, 7, 3]:
        name = "Inception-base-v4"
    else:
        name = "Inception-v4"
        
    model = Model(inputs=inputs, outputs=x, name=name)
    return model


def Inception_v4_backbone(
    num_blocks,
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = Inception_v4(
        num_blocks=num_blocks,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        f"stage1.block{num_blocks[0]}.merger",
        f"stage2.block{num_blocks[1] + 1}.merger",
    ]
    
    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def Inception_base_v4(
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
    
    model = Inception_v4(
        num_blocks=[4, 7, 3],
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


def Inception_base_v4_backbone(
    inputs=[299, 299, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    custom_layers=[]
) -> Model:

    model = Inception_base_v4(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
    )

    custom_layers = custom_layers or [
        "stem.block3",
        "stage1.block2",
        "stage1.block3.merger",
        "stage2.block5.merger",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")
