import tensorflow as tf


class AdaptiveAvgPooling2D(tf.keras.layers.Layer):
    def __init__(self, output_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
        self.data_format = data_format

    def call(self, inputs):
        h_bins, w_bins = self.output_size
        if self.data_format == "channels_last":
            # Chia theo chiều cao
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            # Chia theo chiều rộng
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            # Tính trung bình
            out_vect = tf.reduce_mean(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = tf.reduce_mean(split_rows, axis=[3, 5])
        return out_vect