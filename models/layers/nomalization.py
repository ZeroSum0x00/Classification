import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K


def get_nomalizer_from_name(norm_name, *args, **kwargs):
    norm_name = norm_name.lower()
    if norm_name in ['bn', 'batch', 'batchnorm', 'batch-norm', 'batch norm', 'batch-normalization', 'batch normalization']:
        from tensorflow.keras.layers import BatchNormalization
        return BatchNormalization(*args, **kwargs)
    elif norm_name in ['grn', 'group', 'groupnorm', 'group-norm', 'group norm', 'group-normalization', 'group normalization']:
        from tensorflow.keras.layers import GroupNormalization
        return GroupNormalization(*args, **kwargs)
    elif norm_name in ['ln', 'layer', 'layernorm', 'layer-norm', 'layer norm', 'layer-normalization', 'layer normalization']:
        from tensorflow.keras.layers import LayerNormalization
        return LayerNormalization(*args, **kwargs)
    elif norm_name in ['in', 'instance', 'instancenorm', 'instance-norm', 'instance norm', 'instance-normalization', 'instance normalization']:
        from tensorflow.keras.layers import InstanceNormalization
        return InstanceNormalization(*args, **kwargs)
    elif norm_name in ['sn', 'spectralnorm', 'spectral-norm', 'spectral norm', 'spectral-normalization', 'spectral normalization']:
        from tensorflow.keras.layers import SpectralNormalization
        return SpectralNormalization(*args, **kwargs)
    elif norm_name in ['un', 'unitnorm', 'unit-norm', 'unit norm', 'unit-normalization', 'unit normalization']:
        from tensorflow.keras.layers import UnitNormalization
        return UnitNormalization(*args, **kwargs)
    elif norm_name in ['frn', 'filterresponsenorm', 'filter-response-norm', 'filter response norm', 'filter-response-normalization', 'filter response normalization']:
        from tfa.layers import FilterResponseNormalization
        return FilterResponseNormalization(*args, **kwargs)
    elif norm_name in ['evn', 'evo', 'evonorm', 'evo-norm', 'evo-normalization', 'evo normalization']:
        from models.layers import EvoNormalization
        return EvoNormalization(*args, **kwargs)