import tensorflow as tf

class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, input):
        # Lấy kích thước của input
        input_shape = tf.shape(input)
        
        # Thực hiện phép toán tạo cửa sổ (depth_radius) với tf.image.extract_patches
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, 1, 2 * self.depth_radius + 1, 1],  # Cửa sổ 2*depth_radius+1
            strides=[1, 1, 1, 1],  # Bước nhảy trong chiều không gian
            rates=[1, 1, 1, 1],
            padding='SAME'
        )

        # Tính tổng bình phương các phần tử trong cửa sổ
        sqr_sum = tf.reduce_sum(tf.square(patches), axis=-1, keepdims=True)
        
        # Áp dụng công thức chuẩn hóa cục bộ
        output = input / tf.pow(self.bias + self.alpha * sqr_sum, self.beta)

        return output
