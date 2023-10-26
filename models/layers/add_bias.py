import tensorflow as tf
import tensorflow.keras.backend as K


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, initializer="zeros", axis=-1, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.initializer = initializer
        self.axis        = axis
        
    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            bb_shape = (input_shape[-1],)
        else:
            bb_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                bb_shape[ii] = input_shape[ii]
        self.bb = self.add_weight(name="bias", 
                                  shape=bb_shape, 
                                  initializer=self.initializer, 
                                  trainable=True)
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        return inputs + self.bb

    def get_config(self):
        config = super(BiasLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

    def get_weights_channels_last(self):
        weights = self.get_weights()
        if K.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.squeeze(ii) for ii in weights]
        return weights

    def set_weights_channels_last(self, weights):
        if K.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.reshape(ii, self.bb.shape) for ii in weights]
        return self.set_weights(weights)