import tensorflow as tf
from tensorflow.keras.layers import Activation


class ELSA(tf.keras.layers.Layer):

    def __init__(self, sub_activation="relu", use_elsa=False, alpha=0.9, beta=2.0, **kwargs):
        super(ELSA, self).__init__(**kwargs)
        self.sub_activation = sub_activation
        self.use_elsa = use_elsa
        self.alpha = alpha
        self.beta  = beta
        
    def build(self, input_shape):
        self.activation = Activation(self.sub_activation)
        if self.use_elsa:
            self.alpha = tf.Variable(name="elsa/alpha",
                                     initial_value=[self.alpha],
                                     trainable=True)
            self.beta  = tf.Variable(name="cls_variable",
                                     initial_value=[self.beta],
                                     trainable=True)
        
    def call(self, inputs, training=False):
        if self.use_elsa:
            alpha = tf.clip_by_value(self.alpha, clip_value_min=0.01, clip_value_max=0.99)
            beta  = tf.math.sigmoid(self.beta)
            return self.activation(inputs) + tf.where(tf.greater(inputs, 0), inputs * self.beta, inputs * self.alpha)
        else:
            return self.activation(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
                "sub_activation": self.sub_activation,
                "use_elsa": self.use_elsa,
                "alpha": self.alpha,
                "beta": self.beta
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)