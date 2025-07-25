import tensorflow as tf



class OperatorWrapper(tf.keras.layers.Layer):
    def __init__(self, operator="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = operator

    def call(self, inputs):
        if self.operator.lower() == "sin":
            return tf.math.sin(inputs)
        elif self.operator.lower() == "cos":
            return tf.math.cos(inputs)
        