import tensorflow as tf


class TransposeWrapper(tf.keras.layers.Layer):
    def __init__(self, perm=None, conjugate=False, *args, **kwargs):
        super(TransposeWrapper, self).__init__(*args, **kwargs)
        self.perm = perm
        self.conjugate = conjugate

    def call(self, inputs):
        return tf.transpose(
            inputs,
            perm=self.perm,
            conjugate=self.conjugate,
        )
    