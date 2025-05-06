import tensorflow as tf


class SplitWrapper(tf.keras.layers.Layer):
    def __init__(self, num_or_size_splits, axis=0, num=None, *args, **kwargs):
        super(SplitWrapper, self).__init__(*args, **kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.num = num

    def call(self, inputs):
        return tf.split(
            inputs,
            num_or_size_splits=self.num_or_size_splits,
            axis=self.axis,
            num=self.num,
        )
    