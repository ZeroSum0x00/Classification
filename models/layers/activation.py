import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from .normalization import get_normalizer_from_name


class Mixture(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mixture, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.p = self.add_weight(
            'mixture/p',
            shape       = (1),
            initializer = tf.initializers.Zeros(),
            trainable   = True
        )

    def call(self, inputs, training=False):
        return self.p * inputs + (1 - self.p) * tf.nn.relu(inputs)

        
class HardTanh(tf.keras.layers.Layer):
    def __init__(self, min_val=-1.0, max_val=1.0, **kwargs):
        super(HardTanh, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def call(self, inputs, training=False):
        return tf.clip_by_value(inputs, clip_value_min=self.min_val, clip_value_max=self.max_val)

    def get_config(self):
        config = super().get_config()
        config.update({
                "min_val": self.min_val,
                "max_val": self.max_val,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class HardSwish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.activation = HardTanh(min_val=0.0, max_val=6.0)

    def call(self, inputs, training=False):
        return inputs * self.activation(inputs + 3.0) / 6.0

        
class ReLU6(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.activation = ReLU(max_value=6.)
        
    def call(self, inputs, training=False):
        return self.activation(inputs)


class FReLU(tf.keras.layers.Layer):
    """ FReLU activation https://arxiv.org/abs/2007.11824 """

    def __init__(self, kernel=(3, 3), **kwargs):
        super(FReLU, self).__init__(**kwargs)
        self.kernel = kernel
        
    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.conv = Conv2D(filters=out_dim, 
                           kernel_size=self.kernel, 
                           strides=(1, 1),
                           padding="same",
                           groups=out_dim,
                           use_bias=False)
        self.bn = get_normalizer_from_name('batch-norm')
        
    def call(self, inputs, training=False):
        x = self.conv(inputs, training=training)
        x = self.bn(x, training= training)
        return tf.math.maximum(inputs, x)

    def get_config(self):
        config = super().get_config()
        config.update({
                "kernel": self.kernel,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class AReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.90, beta=2.0, **kwargs):
        super(AReLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta  = beta

    def call(self, inputs, training=None):
        alpha = tf.clip_by_value(self.alpha, clip_value_min=0.01, clip_value_max=0.99)
        beta  = 1 + tf.math.sigmoid(self.beta)
        return tf.nn.relu(inputs) * beta - tf.nn.relu(-inputs) * alpha
      
    def get_config(self):
        config = super().get_config()
        config.update({
                "alpha": self.alpha,
                "beta": self.beta
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))


class MemoryEfficientMish(tf.keras.layers.Layer):
    """  Mish activation memory-efficient """

    def __init__(self, **kwargs):
        super(MemoryEfficientMish, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        if training:
            sx = tf.keras.backend.sigmoid(inputs)
            fx = tf.math.softplus(inputs)
            fx = tf.math.tanh(fx)
            return fx + inputs * sx * (1 - fx * fx)
        else:
            return inputs * tf.math.tanh(tf.math.softplus(inputs))


class SiLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=False):
        return inputs * tf.keras.backend.sigmoid(inputs)

        
class GELUQuick(tf.keras.layers.Layer):
    """https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90-L98
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    
    def __init__(self, **kwargs):
        super(GELUQuick, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        return inputs * tf.keras.backend.sigmoid(1.702 * inputs)


class GELULinear(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GELULinear, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        inputs_abs = tf.math.abs(inputs)
        inputs_sign = tf.math.sign(inputs)
        erf = inputs_abs * -0.7071
        erf = tf.nn.relu(erf + 1.769)
        erf = erf**2 * -0.1444 + 0.5
        return inputs * (erf * inputs_sign + 0.5)
        
        
class AconC(tf.keras.layers.Layer):
    r""" ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    
    def __init__(self, **kwargs):
        super(AconC, self).__init__(**kwargs)

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.p1 = self.add_weight(
            'aconc/p1',
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.p2 = self.add_weight(
            'aconc/p2',
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.beta = self.add_weight(
            'aconc/beta',
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        
    def call(self, inputs, training=False):
        dpx = (self.p1 - self.p2) * inputs
        return dpx * tf.keras.backend.sigmoid(self.beta * dpx) + self.p2 * inputs


class MetaAconC(tf.keras.layers.Layer):
    r""" ACON activation (activate or not)
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    
    def __init__(self, kernel=(1, 1), stride=(1, 1), r=16, **kwargs):
        super(MetaAconC, self).__init__(**kwargs)
        self.kernel = kernel
        self.stride = stride
        self.r = r
        
    def build(self, input_shape):
        out_dim1 = input_shape[-1]
        out_dim2 = max(self.r, out_dim1 // self.r)
        self.p1 = self.add_weight(
            'aconc/p1',
            shape       = (1, 1, 1, out_dim1),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.p2 = self.add_weight(
            'aconc/p2',
            shape       = (1, 1, 1, out_dim1),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        self.conv1 = Conv2D(filters=out_dim2, 
                            kernel_size=self.kernel, 
                            strides=(1, 1),
                            padding="valid",
                            use_bias=True)
        self.conv2 = Conv2D(filters=out_dim1, 
                            kernel_size=self.kernel, 
                            strides=(1, 1),
                            padding="valid",
                            use_bias=True)
        
    def call(self, inputs, training=False):
        y = tf.reduce_mean(inputs, axis=1, keepdims=True)
        y = tf.reduce_mean(y, axis=2, keepdims=True)
        beta = self.conv1(y, training=training)
        beta = self.conv2(beta, training=training)
        beta = tf.keras.backend.sigmoid(beta)
        dpx = (self.p1 - self.p2) * inputs
        return dpx * tf.keras.backend.sigmoid(beta * dpx) + self.p2 * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
                "kernel": self.kernel,
                "stride": self.stride,
                "r": self.r
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ELSA(tf.keras.layers.Layer):

    def __init__(self, sub_activation="relu", use_elsa=False, alpha=0.9, beta=2.0, **kwargs):
        super(ELSA, self).__init__(**kwargs)
        self.sub_activation = sub_activation
        self.use_elsa = use_elsa
        self.alpha = alpha
        self.beta  = beta
        
    def build(self, input_shape):
        self.activation = get_activation_from_name(self.sub_activation)
        if self.use_elsa:
            self.alpha = tf.Variable(name="elsa/alpha",
                                     initial_value=[self.alpha],
                                     trainable=True)
            self.beta  = tf.Variable(name="cls_variable",
                                     initial_value=[self.beta],
                                     trainable=True)
        
    def call(self, inputs, training=False):
        if self.use_elsa:
            alpha = tf.clip_by_value(self.alpha, clip_value_min=0.01, clip_value_max=0.99)
            beta  = tf.math.sigmoid(self.beta)
            return self.activation(inputs) + tf.where(tf.greater(inputs, 0), inputs * self.beta, inputs * self.alpha)
        else:
            return self.activation(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
                "sub_activation": self.sub_activation,
                "use_elsa": self.use_elsa,
                "alpha": self.alpha,
                "beta": self.beta
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_activation_from_name(activ_name, *args, **kwargs):
    activ_name = activ_name.lower()
    if activ_name in ['relu', 'sigmoid', 'softmax', 'softplus', 'phish', 'gelu', 'swish']:
        return Activation(activ_name, *args, **kwargs)
    elif activ_name == 'relu6':
        return ReLU6(*args, **kwargs)
    elif activ_name == 'arelu':
        return AReLU(*args, **kwargs)
    elif activ_name in ['leaky', 'leakyrelu', 'leaky-relu', 'leaky_relu']:
        return LeakyReLU(*args, **kwargs)
    elif activ_name == 'frelu':
        return FReLU(*args, **kwargs)
    elif activ_name == 'mixture':
        return Mixture(*args, **kwargs)
    elif activ_name == 'mish':
        return Mish(*args, **kwargs)
    elif activ_name in ['memoryefficientmish', 'memory-efficient-mish', 'memory_efficient_mish']:
        return MemoryEfficientMish(*args, **kwargs)
    elif activ_name in ['hardtanh', 'hard-tanh', 'hard_tanh']:
        return HardTanh(*args, **kwargs)
    elif activ_name in ['hardwwish', 'hard-swish', 'hard_swish']:
        return HardSwish(*args, **kwargs)
    elif activ_name in ['geluquick', 'gelu-quick', 'gelu_quick']:
        return GELUQuick(*args, **kwargs)
    elif activ_name in ['gelulinear', 'gelu-linear', 'gelu_linear']:
        return GELULinear(*args, **kwargs)
    elif activ_name == 'silu':
        return SiLU(*args, **kwargs)
    elif activ_name == 'aconc':
        return AconC(*args, **kwargs)
    elif activ_name in ['metaaconc', 'meta-aconc', 'meta_aconc']:
        return MetaAconC(*args, **kwargs)
    elif activ_name == 'elsa':
        return ELSA(*args, **kwargs)
    else:
        return Activation('linear')
