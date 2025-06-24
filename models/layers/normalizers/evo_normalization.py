import tensorflow as tf
import tensorflow.keras.backend as K



class EvoNormalization(tf.keras.layers.Layer):
    def __init__(self, nonlinearity=True, num_groups=-1, zero_gamma=False, momentum=0.99, epsilon=0.001, data_format="auto", *args, **kwargs):
        # [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py)
        # EVONORM_B0: nonlinearity=True, num_groups=-1
        # EVONORM_S0: nonlinearity=True, num_groups > 0
        # EVONORM_B0 / EVONORM_S0 linearity: nonlinearity=False, num_groups=-1
        # EVONORM_S0A linearity: nonlinearity=False, num_groups > 0
        super().__init__(*args, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_groups   = num_groups
        self.zero_gamma   = zero_gamma
        self.data_format  = data_format
        self.momentum     = momentum
        self.epsilon      = epsilon
        self.is_channels_first = (
            True if data_format == "channels_first" or (data_format == "auto" and K.image_data_format() == "channels_first") else False
        )

    def build(self, input_shape):
        all_axes = list(range(len(input_shape)))
        param_shape = [1] * len(input_shape)
        if self.is_channels_first:
            param_shape[1] = input_shape[1]
            self.reduction_axes = all_axes[:1] + all_axes[2:]
        else:
            param_shape[-1] = input_shape[-1]
            self.reduction_axes = all_axes[:-1]

        self.gamma = self.add_weight(name="gamma", shape=param_shape, initializer="zeros" if self.zero_gamma else "ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=param_shape, initializer="zeros", trainable=True)
        if self.num_groups <= 0:  # EVONORM_B0
            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                initializer="ones",
                # synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                # aggregation=tf.VariableAggregation.MEAN,
            )
        if self.nonlinearity:
            self.vv = self.add_weight(name="vv", shape=param_shape, initializer="ones", trainable=True)

        if self.num_groups > 0:  # EVONORM_S0
            channels_dim = input_shape[1] if self.is_channels_first else input_shape[-1]
            num_groups = int(self.num_groups)
            while num_groups > 1:
                if channels_dim % num_groups == 0:
                    break
                num_groups -= 1
            self.__num_groups__ = num_groups
            self.groups_dim = channels_dim // self.__num_groups__

            if self.is_channels_first:
                self.group_shape = [-1, self.__num_groups__, self.groups_dim, *input_shape[2:]]
                self.group_reduction_axes = list(range(2, len(self.group_shape)))  # [2, 3, 4]
                self.group_axes = 2
                self.var_shape = [-1, *param_shape[1:]]
            else:
                self.group_shape = [-1, *input_shape[1:-1], self.__num_groups__, self.groups_dim]
                self.group_reduction_axes = list(range(1, len(self.group_shape) - 2)) + [len(self.group_shape) - 1]  # [1, 2, 4]
                self.group_axes = -1
                self.var_shape = [-1, *param_shape[1:]]

    def __group_std__(self, inputs):
        # _group_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L171
        grouped = tf.reshape(inputs, self.group_shape)
        _, var = tf.moments(grouped, self.group_reduction_axes, keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        std = tf.repeat(std, self.groups_dim, axis=self.group_axes)
        return tf.reshape(std, self.var_shape)

    def __batch_std__(self, inputs, training=None):
        # _batch_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L120
        def _call_train_():
            _, var = tf.moments(inputs, self.reduction_axes, keepdims=True)
            # update_op = tf.assign_sub(moving_variance, (moving_variance - variance) * (1 - decay))
            delta = (self.moving_variance - var) * (1 - self.momentum)
            self.moving_variance.assign_sub(delta)
            return var

        def _call_test_():
            return self.moving_variance

        var = K.in_train_phase(_call_train_, _call_test_, training=training)
        return tf.sqrt(var + self.epsilon)

    def __instance_std__(self, inputs):
        # _instance_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L111
        # axes = [1, 2] if data_format == "channels_last" else [2, 3]
        _, var = tf.moments(inputs, self.reduction_axes[1:], keepdims=True)
        return tf.sqrt(var + self.epsilon)

    def call(self, inputs, training=None, **kwargs):
        if self.nonlinearity and self.num_groups > 0:  # EVONORM_S0
            den = self.__group_std__(inputs)
            inputs = inputs * tf.sigmoid(self.vv * inputs) / den
        elif self.num_groups > 0:  # EVONORM_S0a
            # EvoNorm2dS0a https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/evo_norm.py#L239
            den = self.__group_std__(inputs)
            inputs = inputs / den
        elif self.nonlinearity:  # EVONORM_B0
            left = self.__batch_std__(inputs, training)
            right = self.vv * inputs + self.__instance_std__(inputs)
            inputs = inputs / tf.maximum(left, right)
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nonlinearity": self.nonlinearity,
                "zero_gamma": self.zero_gamma,
                "num_groups": self.num_groups,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "data_format": self.data_format,
            }
        )
        return config
    