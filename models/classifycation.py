import numpy as np
import tensorflow as tf
from utils.train_processing import losses_prepare


class CLS(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 name = "SPV", 
                 **kwargs):
        super(CLS, self).__init__(name=name, **kwargs)
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