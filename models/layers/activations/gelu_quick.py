import tensorflow as tf


class GELUQuick(tf.keras.layers.Layer):
    """https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90-L98
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    
    def __init__(self, *args, **kwargs):
        super(GELUQuick, self).__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        return inputs * tf.keras.backend.sigmoid(1.702 * inputs)
    