import tensorflow as tf


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