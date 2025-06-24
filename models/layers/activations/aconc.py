import tensorflow as tf



class AconC(tf.keras.layers.Layer):
    r""" ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        out_dim = input_shape[-1]
        
        self.p1 = self.add_weight(
            "aconc/p1",
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        
        self.p2 = self.add_weight(
            "aconc/p2",
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        
        self.beta = self.add_weight(
            "aconc/beta",
            shape       = (1, 1, 1, out_dim),
            initializer = tf.initializers.RandomNormal(),
            trainable   = True
        )
        
    def call(self, inputs, training=False):
        dpx = (self.p1 - self.p2) * inputs
        return dpx * tf.keras.backend.sigmoid(self.beta * dpx) + self.p2 * inputs
    