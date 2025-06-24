import tensorflow as tf



class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    # def call(self, inputs):
    #     squared = tf.square(inputs)
    
    #     # Expand dims to (B, H, W, C, 1)
    #     squared = tf.expand_dims(squared, axis=-1)
    
    #     # Apply 1D convolution across channels (simulate local depth-wise normalization)
    #     kernel = tf.ones((1, 1, self.depth_radius * 2 + 1, 1, 1))
    #     norm = tf.nn.conv3d(squared, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    #     norm = tf.squeeze(norm, axis=-1)
    #     denom = tf.pow(self.bias + self.alpha * norm, self.beta)
    #     return inputs / denom
    
    def call(self, inputs):
        return tf.nn.local_response_normalization(
            inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "depth_radius": self.depth_radius,
            "bias": self.bias,
            "alpha": self.alpha,
            "beta": self.beta,
        })
        return config
