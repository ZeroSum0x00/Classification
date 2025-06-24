import numpy as np
import tensorflow as tf



@tf.keras.utils.register_keras_serializable()
class CLS(tf.keras.Model):
    def __init__(
        self,
        backbone,
        custom_head=None,
        name="cls",
        *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.backbone = backbone
        self.custom_head = custom_head

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=self.backbone.trainable and training)

        if self.custom_head:
            x = self.custom_head(x, training=self.custom_head.trainable and training)

        if isinstance(x, (list, tuple)):
            return x[-1]
        else:
            return x

    @tf.function
    def predict(self, inputs):
        preds = self(inputs, training=False)
        return preds

    def calc_loss(self, y_true, y_pred, loss_object, sample_weight=None):
        loss = loss_object["loss"]
        loss.coefficient = loss_object["coeff"]
        loss_value = loss(y_true, y_pred, sample_weight=sample_weight)
        return loss_value

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": tf.keras.utils.serialize_keras_object(self.backbone),
            "custom_head": tf.keras.utils.serialize_keras_object(self.custom_head) if self.custom_head else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        backbone_config = config.pop("backbone")
        backbone = tf.keras.utils.deserialize_keras_object(backbone_config)

        custom_head_config = config.pop("custom_head")
        custom_head = tf.keras.utils.deserialize_keras_object(custom_head_config)
        return CLS(backbone=backbone, custom_head=custom_head, **config)
    