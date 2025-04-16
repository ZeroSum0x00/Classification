"""
  # Description:
    - The following table comparing the params of the RepVGG in Tensorflow on 
    size 224 x 224 x 3:

       ----------------------------------------------------------------------
      |      Model Name       |    Un-deploy params    |    Deploy params    |
      |----------------------------------------------------------------------|
      |       RepVGG-A0       |         9,132,616      |      8,309,384      |
      |----------------------------------------------------------------------|
      |       RepVGG-A1       |         14,122,088     |     12,789,864      |
      |----------------------------------------------------------------------|
      |       RepVGG-A2       |         28,253,160     |     25,499,944      |
      |----------------------------------------------------------------------|
      |       RepVGG-B0       |         15,853,160     |     14,339,048      |
      |----------------------------------------------------------------------|
      |       RepVGG-B1       |         57,483,112     |     51,829,480      |
      |----------------------------------------------------------------------|
      |      RepVGG-B1g2      |         45,850,472     |     41,360,104      |
      |----------------------------------------------------------------------|
      |      RepVGG-B1g4      |         40,034,152     |     36,125,416      |
      |----------------------------------------------------------------------|
      |       RepVGG-B2       |         89,107,432     |     80,315,112      |
      |----------------------------------------------------------------------|
      |      RepVGG-B2g2      |         70,931,432     |     63,956,712      |
      |----------------------------------------------------------------------|
      |      RepVGG-B2g4      |         61,843,432     |     55,777,512      |
      |----------------------------------------------------------------------|
      |       RepVGG-B3       |        123,185,256     |    110,960,872      |
      |----------------------------------------------------------------------|
      |      RepVGG-B3g2      |         97,011,816     |     87,404,776      |
      |----------------------------------------------------------------------|
      |      RepVGG-B3g4      |         83,925,096     |     75,626,728      |
       ----------------------------------------------------------------------

  # Reference:
    - [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)
    - Source: https://github.com/hoangthang1607/RepVGG-Tensorflow-2/tree/main

"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, ZeroPadding2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from tensorflow.keras.regularizers import l2

from models.layers import (
    LinearLayer, AdaptiveAvgPooling2D,
    get_activation_from_name, get_normalizer_from_name
)
from utils.model_processing import process_model_input, compute_padding



optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


class SEBlock(tf.keras.layers.Layer):
    
    def __init__(
        self,
        expansion=0.5,
        activation="relu",
        normalizer=None,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        *args, **kwargs
    ):
        super(SEBlock, self).__init__()
        self.expansion = expansion
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        
    def build(self, input_shape):
        bs = input_shape[-1]
        hidden_dim = int(bs * self.expansion)
        
        self.conv1 = Sequential([
            Conv2D(
                filters=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=l2(self.regularizer_decay),
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name(self.activation),
        ])
        
        self.conv2 = Sequential([
            Conv2D(
                filters=bs,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=l2(self.regularizer_decay),
            ),
            get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps),
            get_activation_from_name('hard-sigmoid'),
        ])
        
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.avg_pool(inputs)
        x = self.conv1(x, training=training)        
        x = self.conv2(x, training=training)
        return x * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "expansion": self.expansion,
            "activation": self.activation,
            "normalizer": self.normalizer,
        })
        return config


class RepVGGBlock(tf.keras.layers.Layer):

    '''
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        dilation_rate=(1, 1),
        groups=1,
        use_se=False,
        activation='relu',
        normalizer='batch-norm',
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        deploy=False,
        *args, **kwargs
    ):
        super(RepVGGBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, tuple) else (dilation_rate, dilation_rate)
        self.groups = groups
        self.use_se = use_se
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = regularizer_decay
        self.norm_eps = norm_eps
        self.deploy = deploy

        if padding.lower() not in ["same", "valid"]:
            raise ValueError(f"Invalid padding type: {padding}. Expected 'same' or 'valid'.")
        self.padding_mode = padding.lower()

        if self.padding_mode == 'valid' and not self.deploy:
            print(f"[Warning] Using padding='valid' in training mode requires manual shape matching!")

    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        if self.padding_mode == "same":
            pad_h, pad_w = compute_padding(self.kernel_size, self.dilation_rate)
            pad_1x1_h, pad_1x1_w = compute_padding((1, 1), self.dilation_rate)
        else:
            pad_h, pad_w = 0, 0
            pad_1x1_h, pad_1x1_w = 0, 0

        self.nonlinearity = get_activation_from_name(self.activation, name=self.name + '_nonlinearity')

        if self.use_se:
            self.se_block = SEBlock(1/16)
        else:
            self.se_block = LinearLayer()

        if self.deploy:
            self.rbr_reparam = Sequential([
                    ZeroPadding2D(padding=(pad_h, pad_w)),
                    Conv2D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding="valid",
                        dilation_rate=self.dilation_rate,
                        groups=self.groups,
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=l2(self.regularizer_decay),
                    )
            ], name=self.name + '_rbr_reparam')
        else:
            self.rbr_identity = get_normalizer_from_name(
                self.normalizer,
                epsilon=self.norm_eps,
                name=self.name + '_rbr_identity'
            ) if self.filters == self.in_channels and self.strides == (1, 1) else None

            self.rbr_dense = self.convolution_block(filters=self.filters,
                                                    kernel_size=self.kernel_size,
                                                    strides=self.strides,
                                                    padding=(pad_h, pad_w),
                                                    dilation_rate=self.dilation_rate,
                                                    groups=self.groups,
                                                    name=self.name + '_rbr_dense'
            )
            self.rbr_1x1 = self.convolution_block(filters=self.filters,
                                                  kernel_size=(1, 1),
                                                  strides=self.strides,
                                                  padding=(pad_1x1_h, pad_1x1_w),
                                                  dilation_rate=self.dilation_rate,
                                                  groups=self.groups,
                                                  name=self.name + '_rbr_1x1'
            )

    def convolution_block(self, filters, kernel_size, strides, padding, dilation_rate=(1, 1), groups=1, name=None):
        return Sequential([
                ZeroPadding2D(padding=padding, name="reppadding"),
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    dilation_rate=dilation_rate,
                    groups=groups,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=l2(self.regularizer_decay),
                    name="repconv"
                ),
                get_normalizer_from_name(
                    self.normalizer,
                    epsilon=self.norm_eps,
                    name="repbn"
                ),
        ], name=name
        )

    def _match_shape(self, x, target_shape):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        target_h, target_w = target_shape[0], target_shape[1]
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return x[:, offset_h:offset_h+target_h, offset_w:offset_w+target_w, :]

    def call(self, inputs, training=False):
        if self.deploy:
            x = self.rbr_reparam(inputs, training=training)
            x = self.se_block(x, training=training)
            return self.nonlinearity(x)

        x_dense = self.rbr_dense(inputs, training=training)
        x_1x1 = self.rbr_1x1(inputs, training=training)

        if self.padding_mode == 'valid':
            target_shape = tf.shape(x_dense)[1:3]

            x_1x1 = self._match_shape(x_1x1, target_shape)

            if self.rbr_identity is not None:
                id_out = self.rbr_identity(inputs, training=training)
                id_out = self._match_shape(id_out, target_shape)
            else:
                id_out = 0
        else:
            id_out = self.rbr_identity(inputs, training=training) if self.rbr_identity else 0

        x = x_dense + x_1x1 + id_out
        x = self.se_block(x, training=training)
        return self.nonlinearity(x)


    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    #     for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(
                kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]])
            )

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, Sequential):
            kernel = branch.get_layer("repconv").weights[0]
            running_mean = branch.get_layer("repbn").moving_mean
            running_var = branch.get_layer("repbn").moving_variance
            gamma = branch.get_layer("repbn").gamma
            beta = branch.get_layer("repbn").beta
            eps = branch.get_layer("repbn").epsilon
        else:
            assert isinstance(branch, BatchNormalization)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, input_dim, self.in_channels), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[1, 1, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(
                    kernel_value, dtype=np.float32
                )
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        std = tf.sqrt(running_var + eps)
        t = gamma / std
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias


def RepVGG(
    num_blocks,
    width_multiplier=None,
    override_groups_map=None,
    use_se=False,
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
    drop_rate=0.1,
    deploy=False,
):

    if weights not in {"imagenet", None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == "imagenet" and include_head and num_classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_head`'
                         ' as true, `num_classes` should be 1000')

    assert len(width_multiplier) == 4
    override_groups_map = override_groups_map or dict()
    assert 0 not in override_groups_map

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

    current_layer_idx = 0
               
    x = RepVGGBlock(
        filters=min(64, int(64 * width_multiplier[0])),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="SAME",
        use_se=use_se,
        deploy=deploy,
        **layer_constant_dict,
        name=f"stem"
    )(inputs)
    current_layer_idx += 1

    for i in range(num_blocks[0]):
        cur_groups = override_groups_map.get(current_layer_idx, 1)
        x = RepVGGBlock(
            filters=int(64 * width_multiplier[0]),
            kernel_size=3,
            strides=(2, 2) if i == 0 else (1, 1),
            padding="SAME",
            groups=cur_groups,
            use_se=use_se,
            deploy=deploy,
            **layer_constant_dict,
            name=f"stage1.block{i + 1}"
        )(x)
        current_layer_idx += 1

    for i in range(num_blocks[1]):
        cur_groups = override_groups_map.get(current_layer_idx, 1)
        x = RepVGGBlock(
            filters=int(128 * width_multiplier[1]),
            kernel_size=3,
            strides=(2, 2) if i == 0 else (1, 1),
            padding="SAME",
            groups=cur_groups,
            use_se=use_se,
            deploy=deploy,
            **layer_constant_dict,
            name=f"stage2.block{i + 1}"
        )(x)
        current_layer_idx += 1

    for i in range(num_blocks[2]):
        cur_groups = override_groups_map.get(current_layer_idx, 1)
        x = RepVGGBlock(
            filters=int(256 * width_multiplier[2]),
            kernel_size=3,
            strides=(2, 2) if i == 0 else (1, 1),
            padding="SAME",
            groups=cur_groups,
            use_se=use_se,
            deploy=deploy,
            **layer_constant_dict,
            name=f"stage3.block{i + 1}"
        )(x)
        current_layer_idx += 1

    for i in range(num_blocks[3]):
        cur_groups = override_groups_map.get(current_layer_idx, 1)
        x = RepVGGBlock(
            filters=int(512 * width_multiplier[3]),
            kernel_size=3,
            strides=(2, 2) if i == 0 else (1, 1),
            padding="SAME",
            groups=cur_groups,
            **layer_constant_dict,
            use_se=use_se,
            deploy=deploy,
            name=f"stage4.block{i + 1}"
        )(x)

    if include_head:
        x = AdaptiveAvgPooling2D(output_size=1)(x)
        x = Flatten(name='flatten')(x)
        x = Dropout(drop_rate)(x)
        x = Dense(1 if num_classes == 2 else num_classes, name='predictions')(x)
        x = get_activation_from_name(final_activation)(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    if num_blocks == [2, 4, 14, 1]:
        if width_multiplier == [0.75, 0.75, 0.75, 2.5]:
            i = 0
        elif width_multiplier == [1, 1, 1, 2.5]:
            i = 1
        elif width_multiplier == [1.5, 1.5, 1.5, 2.75]:
            i = 2
        else:
            i = ''
        model = Model(inputs=inputs, outputs=x, name=f'RepVGG-A{i}')
        
    elif num_blocks == [4, 6, 16, 1]:
        if width_multiplier == [1, 1, 1, 2.5]:
            i = 0
        elif width_multiplier == [2, 2, 2, 4]:
            i = 1
        elif width_multiplier == [2.5, 2.5, 2.5, 5]:
            i = 2
        elif width_multiplier == [3, 3, 3, 5]:
            i = 3
        else:
            i = ''

        if override_groups_map == g2_map:
            g = 2
        elif override_groups_map == g4_map:
            g = 4
        else:
            g = ''

        if g != '':
            model = Model(inputs=inputs, outputs=x, name=f'RepVGG-B{i}g{g}')
        else:
            model = Model(inputs=inputs, outputs=x, name=f'RepVGG-B{i}')
    else:
        model = Model(inputs=inputs, outputs=x, name='RepVGG')

    return model


def RepVGG_backbone(
    num_blocks,
    width_multiplier,
    override_groups_map=None,
    use_se=False,
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG(
        num_blocks=num_blocks,
        width_multiplier=width_multiplier,
        override_groups_map=override_groups_map,
        use_se=use_se,
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        f"stage1.block{num_blocks[0]}",
        f"stage2.block{num_blocks[1]}",
        f"stage3.block{num_blocks[2]}",
    ]
    print(custom_layers)

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")


def RepVGG_A0(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:

    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_A0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_A0(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_A1(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:

    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_A1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_A1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_A2(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_A2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_A2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block2",
        "stage2.block4",
        "stage3.block14",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B0(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B0_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B0(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B1(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B1_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B1(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B1g2(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B1g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B1g2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B1g4(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B1g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B1g4(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B2(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B2g2(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B2g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B2g2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B2g4(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B2g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B2g4(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B3(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B3_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B3(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B3g2(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B3g2_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B3g2(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def RepVGG_B3g4(
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
    drop_rate=0.1,
    deploy=False,
) -> Model:
    
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        use_se=False,
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
        deploy=deploy,
    )
    return model


def RepVGG_B3g4_backbone(
    inputs=[224, 224, 3],
    weights="imagenet",
    activation="relu",
    normalizer="batch-norm",
    deploy=False,
    custom_layers=[]
) -> Model:

    model = RepVGG_B3g4(
        inputs=inputs,
        include_head=False,
        weights=weights,
        activation=activation,
        normalizer=normalizer,
        deploy=deploy,
    )
    
    custom_layers = custom_layers or [
        "stem",
        "stage1.block4",
        "stage2.block6",
        "stage3.block16",
    ]

    outputs = [model.get_layer(layer).output for layer in custom_layers]
    final_output = model.get_layer(model.layers[-1].name).output
    return Model(inputs=model.inputs, outputs=[*outputs, final_output], name=f"{model.name}_backbone")

    
def repvgg_reparameter(model: tf.keras.Model, structure, input_shape=(224, 224, 3), classes=1000, save_path=None):
    for layer, deploy_layer in zip(model.layers, structure.layers):
        if hasattr(layer, "deploy") and layer.deploy == True:
            print(f'layer {layer.name} is deployed, passing reparameter.')
            continue

        if hasattr(layer, "repvgg_convert"):
            kernel, bias = layer.repvgg_convert()
            deploy_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.Sequential):
            assert isinstance(deploy_layer, tf.keras.Sequential)
            for sub_layer, deploy_sub_layer in zip(layer.layers, deploy_layer.layers):
                kernel, bias = sub_layer.repvgg_convert()
                deploy_sub_layer.rbr_reparam.layers[1].set_weights([kernel, bias])
        elif isinstance(layer, tf.keras.layers.Dense):
            assert isinstance(deploy_layer, tf.keras.layers.Dense)
            weights = layer.get_weights()
            deploy_layer.set_weights(weights)

    if save_path is not None:
        structure.save_weights(save_path)

    return structure