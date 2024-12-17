from .relu6 import ReLU6
from .arelu import AReLU
from .frelu import FReLU
from .mixture import Mixture
from .mish import Mish
from .memory_efficient_mish import MemoryEfficientMish
from .hardtanh import HardTanh
from .hardswish import HardSwish
from .gelu_linear import GELULinear
from .gelu_quick import GELUQuick
from .silu import SiLU
from .aconc import AconC
from .meta_aconc import MetaAconC
from .elsa import ELSA
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU


def get_activation_from_name(activ_name, *args, **kwargs):
    if activ_name:
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
        elif activ_name in ['hardswish', 'hard-swish', 'hard_swish']:
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
    else:
        return Activation('linear')