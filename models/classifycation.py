import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense



@tf.keras.utils.register_keras_serializable()
class CLS(tf.keras.Model):
    def __init__(
        self,
        backbone,
        custom_head=None,
        num_classes=1000,
        name="CLS",
        *args, **kwargs
    ):
        super(CLS, self).__init__(*args, **kwargs, name=name)
        self.backbone = backbone
        self.custom_head = custom_head
        self.num_classes = num_classes

    def build(self, input_shape):
        if self.custom_head:
            try:
                latted_dim = self.custom_head.output.shape
            except:
                latted_dim = self.custom_head.output_shape
        else:
            features_shape = self.backbone.output
            if isinstance(features_shape, (list, tuple)):
                latted_dim = features_shape[-2].shape
            else:
                latted_dim = features_shape.shape

        if len(latted_dim) > 2:
            self.flat_layer = Reshape(target_shape=(-1,))

        if latted_dim[-1] != self.num_classes:
            self._head = Dense(
                units=self.num_classes,
                activation="sigmoid" if self.num_classes == 2 else "softmax",
                name="classifier_head"
        )
        
    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=self.backbone.trainable and training)

        x = features[-1] if isinstance(features, (list, tuple)) else features

        if self.custom_head:
            x = self.custom_head(x, training=self.custom_head.trainable and training)
        
        if hasattr(self, "flat_layer"):
            x = self.flat_layer(x)

        if hasattr(self, "_head"):            
            x = self._head(x, training=training)

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
    