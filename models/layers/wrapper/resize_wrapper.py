import tensorflow as tf



class ResizeWrapper(tf.keras.layers.Layer):
    def __init__(
        self,
        size,
        method="bilinear",
        preserve_aspect_ratio=False,
        antialias=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.size = size
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias
        if self.method not in ["bilinear", "lanczos3", "lanczos5", "bicubic", "gaussian", "nearest", "area", "mitchellcubic"]:
            raise ValueError("Invalid method")
            
    def call(self, inputs):
        return tf.image.resize(
            inputs,
            size=self.size,
            method=self.method,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            antialias=self.antialias,
        )
    