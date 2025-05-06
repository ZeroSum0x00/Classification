import tensorflow as tf

class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, *args, **kwargs):
        super(LocalResponseNormalization, self).__init__(*args, **kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, input):
        input_shape = tf.shape(input)
        
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, 1, 2 * self.depth_radius + 1, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME"
        )

        sqr_sum = tf.reduce_sum(tf.square(patches), axis=-1, keepdims=True)
        
        output = input / tf.pow(self.bias + self.alpha * sqr_sum, self.beta)
        return output
