import tensorflow as tf



class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, units, r=4, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.r = r
        self.alpha = alpha

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=[input_dim, self.units],
            initializer="glorot_uniform",
            trainable=False,
            name="W"
        )
        
        self.A = self.add_weight(
            shape=[input_dim, self.r],
            initializer="random_normal",
            trainable=True,
            name="A"
        )
        
        self.B = self.add_weight(
            shape=[self.r, self.units],
            initializer="zeros",
            trainable=True,
            name="B"
        )
        
    def call(self, inputs):
        lora_out = tf.matmul(inputs, self.A)
        lora_out = tf.matmul(lora_out, self.B) * (self.alpha / self.r)
        return tf.matmul(inputs, self.W) + lora_out
