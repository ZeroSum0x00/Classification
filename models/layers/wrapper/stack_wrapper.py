import tensorflow as tf



class StackWrapper(tf.keras.layers.Layer):
    def __init__(self, axis=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)


class UnstackWrapper(tf.keras.layers.Layer):
    def __init__(self, num=None, axis=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num = num
        self.axis = axis

    def call(self, inputs):
        return tf.unstack(inputs, num=self.num, axis=self.axis)
    