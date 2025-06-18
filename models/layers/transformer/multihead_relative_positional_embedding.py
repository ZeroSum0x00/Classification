import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class MultiHeadRelativePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_heads=-1, attn_height=-1, cls_token=True, *args, **kwargs):
        super(MultiHeadRelativePositionalEmbedding, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.attn_height = attn_height
        self.cls_token = cls_token

        if cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        if self.attn_height == -1:
            height = width = int(float(attn_shape[2] - self.cls_token_len) ** 0.5)
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
            
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        
        self.positional_embedding = self.add_weight(
            shape=[num_heads, num_relative_distance],
            initializer="zeros",
            trainable=True,
            name="positional_embedding"
        )

        hh, ww = np.meshgrid(range(height), range(width))
        coords = np.stack([hh, ww], axis=-1)
        coords_flatten = np.reshape(coords, [-1, 2])
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = np.sum(relative_coords, axis=-1).astype("float32")
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.cls_token:
            top = np.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = np.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = np.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            left_corner = np.concatenate([corner, left], axis=0)
            relative_position_index = np.concatenate([top, relative_position_index], axis=0)
            relative_position_index = np.concatenate([left_corner, relative_position_index], axis=1)

        # relative_position_index = tf.convert_to_tensor(relative_position_index, dtype="int64")
        self.relative_position_index = self.add_weight(
            shape=relative_position_index.shape,
            initializer=relative_position_index,
            trainable=False,
            dtype=tf.int64,
            name="relative_position_index"
        )
        super().build(attn_shape)

    def call(self, inputs, training=False):
        relative_position_mask = self.relative_position_index[: inputs.shape[2], : inputs.shape[3]]
        pos_emb = tf.gather(self.positional_embedding, relative_position_mask, axis=1)
        return inputs + pos_emb

    def load_resized_weights(self, source_layer, method="bilinear"):
        if isinstance(source_layer, dict):
            source_tt = list(source_layer.values())[0]
        else:
            source_tt = source_layer.get_weights()[0]
            
        source_tt = np.array(source_tt).astype("float32")
        hh = ww = int(float(source_tt.shape[1] - self.cls_token_pos_len) ** 0.5)
        num_heads = source_tt.shape[0]
        ss = source_tt[:, : hh * ww].reshape((num_heads, hh, ww))

        if self.attn_height == -1:
            target_hh = target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) / target_hh)

        tt = K.numpy_image_resize(ss, target_shape=[target_hh, target_ww], method=method, is_source_channels_last=False)
        tt = tt.reshape((num_heads, tt.shape[1] * tt.shape[2]))
        
        if self.cls_token:
            tt = np.concatenate([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
            
        self.set_weights([tt])

    def show_pos_emb(self, rows=1, base_size=2):
        import math
        import matplotlib.pyplot as plt

        num_heads = self.positional_embedding.shape[0]
        hh = ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        pos_emb = self.positional_embedding[:, : hh * ww]
        pos_emb = pos_emb.numpy() if hasattr(pos_emb, "numpy") else np.array(pos_emb)
        pos_emb = pos_emb.reshape((num_heads, hh, ww))
        cols = int(math.ceil(num_heads / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            if id >= num_heads:
                break
            ax.imshow(pos_emb[id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "attn_height": self.attn_height,
            "cls_token": self.cls_token
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    