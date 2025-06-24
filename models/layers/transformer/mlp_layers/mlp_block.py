import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout

from models.layers import get_normalizer_from_name, get_activation_from_name
from utils.model_processing import check_regularizer



class MLPBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        mlp_dim,
        out_dim=-1,
        use_conv=False,
        use_bias=True,
        use_gated=False,
        activation="gelu",
        normalizer=None,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        regularizer_decay=5e-4,
        norm_eps=1e-6,
        drop_rate=0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.use_conv = use_conv
        self.use_bias = use_bias
        self.use_gated = use_gated
        self.activation = activation
        self.normalizer = normalizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_decay = check_regularizer(regularizer_decay)
        self.norm_eps = norm_eps
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        hidden_dim = self.mlp_dim * 2 if self.use_gated else self.mlp_dim
        if not self.use_conv:
            self.linear1 = Dense(
                units=hidden_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )
            
            self.linear2 = Dense(
                units=self.out_dim if self.out_dim > 0 else input_shape[-1],
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )
            
        else:
            self.linear1 = Conv2D(
                filters=hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )
            
            self.linear2 = Conv2D(
                filters=self.out_dim if self.out_dim > 0 else input_shape[-1],
                kernel_size=(1, 1),
                strides=(1, 1),
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.regularizer_decay,
            )
            
        if self.normalizer:
            self.norm = get_normalizer_from_name(self.normalizer, epsilon=self.norm_eps)
            
        self.activation = get_activation_from_name(self.activation)
        self.dropout = Dropout(self.drop_rate)
        
    def call(self, inputs, training=False):
        x = self.linear1(inputs, training=training)
        
        if self.use_gated:
            gate, x = tf.split(x, 2, axis=-1)
            gate = self.activation(gate)
            x = gate * x
        else:
            x = self.activation(x)
            
        x = self.dropout(x, training=training)
        
        if self.normalizer:
            x = self.norm(x, training=training)
            
        x = self.linear2(x, training=training)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "mlp_dim": self.mlp_dim,
            "out_dim": self.out_dim,
            "use_conv": self.use_conv,
            "use_bias": self.use_bias,
            "use_gated": self.use_gated,
            "activation": self.activation,
            "normalizer": self.normalizer,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "regularizer_decay": self.regularizer_decay,
            "norm_eps": self.norm_eps,
            "drop_rate": self.drop_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        