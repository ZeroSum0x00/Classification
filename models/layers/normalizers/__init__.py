import tensorflow as tf
from tensorflow.keras import backend as K
from ..linear_layer import LinearLayer


def get_normalizer_from_name(norm_name, *args, **kwargs):
    if norm_name:
        norm_name = norm_name.lower()
        if norm_name in ["bn", "batch", "batchnorm", "batch-norm", "batch norm", "batch-normalization", "batch normalization"]:
            from tensorflow.keras.layers import BatchNormalization
            return BatchNormalization(*args, **kwargs)
        elif norm_name in ["grn", "group", "groupnorm", "group-norm", "group norm", "group-normalization", "group normalization"]:
            from tensorflow.keras.layers import GroupNormalization
            return GroupNormalization(*args, **kwargs)
        elif norm_name in ["ln", "layer", "layernorm", "layer-norm", "layer norm", "layer-normalization", "layer normalization"]:
            from tensorflow.keras.layers import LayerNormalization
            return LayerNormalization(*args, **kwargs)
        elif norm_name in ["in", "instance", "instancenorm", "instance-norm", "instance norm", "instance-normalization", "instance normalization"]:
            from tensorflow.keras.layers import InstanceNormalization
            return InstanceNormalization(*args, **kwargs)
        elif norm_name in ["sn", "spectralnorm", "spectral-norm", "spectral norm", "spectral-normalization", "spectral normalization"]:
            from tensorflow.keras.layers import SpectralNormalization
            return SpectralNormalization(*args, **kwargs)
        elif norm_name in ["un", "unitnorm", "unit-norm", "unit norm", "unit-normalization", "unit normalization"]:
            from tensorflow.keras.layers import UnitNormalization
            return UnitNormalization(*args, **kwargs)
        elif norm_name in ["evn", "evo", "evonorm", "evo-norm", "evo-normalization", "evo normalization"]:
            from .evo_normalization import EvoNormalization
            return EvoNormalization(*args, **kwargs)
        elif norm_name in ["local-response", "local response", "local-response-norm", "local response norm", "local-response-normalization", "local response normalization"]:
            from .local_response_normalization import LocalResponseNormalization
            return LocalResponseNormalization(*args, **kwargs)
            
    else:
        return LinearLayer(*args, **kwargs)
    