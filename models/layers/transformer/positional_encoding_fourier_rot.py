import numpy as np
import tensorflow as tf



class PositionalEncodingFourierRot1D(tf.keras.layers.Layer):
    def __init__(self, max_block_size, temperature=1e4, *args, **kwargs):
        super(PositionalEncodingFourierRot1D, self).__init__(*args, **kwargs)
        self.max_block_size = max_block_size
        self.temperature = float(temperature)

    def build(self, input_shape):
        self.channels = input_shape[-2] * input_shape[-1]
        pos_filters = self.channels // 2
        dim_t = self.temperature ** (np.arange(pos_filters, dtype="float32") / pos_filters)
        grid = np.expand_dims(np.arange(self.max_block_size, dtype="float32"), -1) / dim_t
        pos_sin, pos_cos = np.expand_dims(np.sin(grid), -2), np.expand_dims(np.cos(grid), -2)
        self.pos_sin = tf.convert_to_tensor(pos_sin)
        self.pos_cos = tf.convert_to_tensor(pos_cos)
        super().build(input_shape)

    def call(self, inputs, training=False):
        left, right = tf.unstack(inputs, axis=-1)
        pos_cos = self.pos_cos[: left.shape[-3]]
        pos_sin = self.pos_sin[: left.shape[-3]]
        out = tf.stack([left * pos_cos - right * pos_sin, right * pos_cos + left * pos_sin], axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_block_size": self.max_block_size,
            "temperature": self.temperature
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class PositionalEncodingFourierRot(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads=-1,
        attn_height=-1,
        cls_token=True,
        temperature=1e4,
        ref_feature_shape=16,
        *args, **kwargs
    ):
        super(PositionalEncodingFourierRot, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.attn_height = attn_height
        self.cls_token = cls_token
        self.temperature = float(temperature)
        self.ref_feature_shape = ref_feature_shape
        self.cls_token_len = 1 if cls_token else 0

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.attn_height == -1:
            height = width = int(float(input_shape[-2] - self.cls_token_len) ** 0.5)
        else:
            height = self.attn_height
            width = int(float(input_shape[-2] - self.cls_token_len) / height)
        self.blocks_shape = [*input_shape[1:-2], input_shape[-2] - self.cls_token_len]

        hh = np.arange(height, dtype="float32")
        ww = np.arange(width, dtype="float32")
        
        if self.ref_feature_shape is not None and self.ref_feature_shape > 0:
            hh = hh / height * self.ref_feature_shape
            ww = ww / height * self.ref_feature_shape

        pos_fileters = (self.channels // self.num_heads // 4) if self.num_heads > 0 else (self.channels // 4)
        dim_t = self.temperature ** (np.arange(pos_fileters, dtype="float32") / pos_fileters)
        grid = np.stack(np.meshgrid(hh, ww, indexing="ij"), axis=-1)
        grid = np.expand_dims(grid, -1) / dim_t
        grid = np.reshape(grid, [height, width, -1])
        pos_sin, pos_cos = np.sin(grid), np.cos(grid)
        pos_sin, pos_cos = np.repeat(pos_sin, 2, axis=-1), np.repeat(pos_cos, 2, axis=-1)

        if self.num_heads > 0:
            pos_sin = np.repeat(np.expand_dims(pos_sin, axis=-2), self.num_heads, axis=-2).reshape([height * width, self.num_heads * pos_fileters * 4])
            pos_cos = np.repeat(np.expand_dims(pos_cos, axis=-2), self.num_heads, axis=-2).reshape([height * width, self.num_heads * pos_fileters * 4])
        else:
            pos_sin = np.reshape(pos_sin, [height * width, pos_fileters * 4])
            pos_cos = np.reshape(pos_cos, [height * width, pos_fileters * 4])

        self.pos_sin = tf.convert_to_tensor(pos_sin)
        self.pos_cos = tf.convert_to_tensor(pos_cos)
        super().build(input_shape)

    def call(self, inputs, training=False):
        if self.cls_token:
            cls_token, inputs = tf.split(inputs, [1, -1], axis=-2)

        left, right = tf.split(tf.reshape(inputs, [-1, *self.blocks_shape, self.channels // 2, 2]), 2, axis=-1)
        rot = tf.concat([-right, left], axis=-1)
        rot = tf.reshape(rot, (-1, *self.blocks_shape, self.channels))
        out = inputs * self.pos_cos + rot * self.pos_sin

        if self.cls_token:
            out = tf.concat([cls_token, out], axis=-2)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "attn_height": self.attn_height,
            "cls_token": self.cls_token,
            "temperature": self.temperature,
            "ref_feature_shape": self.ref_feature_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    