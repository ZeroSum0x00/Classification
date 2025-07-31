import tensorflow as tf



class MaxPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="VALID", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.upper()

    def call(self, inputs):
        return tf.nn.pool(
            input=inputs,
            window_shape=self.pool_size,
            pooling_type="MAX",
            strides=self.strides,
            padding=self.padding,
            data_format="NHWC"
        )


class MaxPool2D_Scratch(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding.upper()

    def call(self, inputs):
        batch_size, h, w, c = tf.unstack(tf.shape(inputs))
        ksize_y, ksize_x = self.pool_size
        stride_y, stride_x = self.strides

        # extract_patches: [B, out_h, out_w, ksize_y * ksize_x * C]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, ksize_y, ksize_x, 1],
            strides=[1, stride_y, stride_x, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )

        # Reshape patches to [B, out_h, out_w, ksize_y * ksize_x, C]
        # Step 1: group per-channel
        patch_dim = ksize_y * ksize_x
        patches = tf.reshape(patches, [batch_size, -1, -1, patch_dim, c])

        # Step 2: max over patch dimension
        output = tf.reduce_max(patches, axis=3)
        return output
