import numpy as np
import tensorflow as tf
from utils.train_processing import losses_prepare


class CLS(tf.keras.Model):
    def __init__(self, backbone, name = "CLS", *args, **kwargs):
        super(CLS, self).__init__(*args, **kwargs)
        self.backbone = backbone

    def call(self, inputs, training=False):
        feature_maps = self.backbone(inputs, training=training)
        
        if isinstance(feature_maps, (list, tuple)):
            return feature_maps[-1]
        else:
            return feature_maps

    @tf.function
    def predict(self, inputs):
        preds = self(inputs, training=False)
        return preds

    def calc_loss(self, y_true, y_pred, loss_object):
        loss = losses_prepare(loss_object)
        loss_value = 0
        if loss:
            loss_value += loss(y_true, y_pred)
        return loss_value

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": self.backbone.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        from tensorflow.keras.models import Model
        
        backbone_config = config.pop("backbone")
        backbone = Model.from_config(backbone_config)
        return CLS(backbone=backbone, **config)
