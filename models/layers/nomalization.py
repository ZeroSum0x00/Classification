import tensorflow as tf
from tensorflow.keras import backend as K


def get_nomalizer_from_name(name, *args, **kwargs):
    norm_name = name.lower()
    if norm_name in ['bn', 'batch', 'batchnorm', 'batch-norm', 'batch norm', 'batch-normalization', 'batch normalization']:
        from tensorflow.keras.layers import BatchNormalization
        return BatchNormalization(*args, **kwargs)
    elif norm_name in ['gr', 'group', 'groupnorm', 'group-norm', 'group norm', 'group-normalization', 'group normalization']:
        from tensorflow.keras.layers import GroupNormalization
        return GroupNormalization(*args, **kwargs)
    elif norm_name in ['layer', 'layernorm', 'layer-norm', 'layer norm', 'layer-normalization', 'layer normalization']:
        from tensorflow.keras.layers import LayerNormalization
        return LayerNormalization(*args, **kwargs)
    elif norm_name in ['evo', 'evonorm', 'evo-norm', 'evo-normalization', 'evo normalization']:
        from models.layers import EvoNormalization
        return EvoNormalization(*args, **kwargs)