import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape


@tf.keras.utils.register_keras_serializable()
class CLS(tf.keras.Model):

    def __init__(
        self,
        backbone,
        custom_head=None,
        num_classes=1000,
        name="CLS",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.backbone = backbone
        self.custom_head = custom_head
        self.num_classes = num_classes

        self.flat_layer = None
        self.classifier_head = None

        self._saved_input_shape = None

    def build(self, input_shape):

        self._saved_input_shape = tuple(input_shape)

        # Build backbone nếu cần
        if not self.backbone.built:
            self.backbone.build(input_shape)

        # Dummy tensor để suy shape thực tế
        dummy = tf.zeros(
            [1] + list(input_shape[1:]),
            dtype=tf.float32
        )

        x = self.backbone(dummy, training=False)

        if isinstance(x, (list, tuple)):
            x = x[-1]

        if self.custom_head is not None:

            if not self.custom_head.built:
                self.custom_head.build(x.shape)

            x = self.custom_head(x, training=False)

        feature_shape = x.shape
        rank = feature_shape.rank

        # Flatten nếu output > 2 chiều
        if rank is not None and rank > 2:

            self.flat_layer = Reshape(
                (-1,),
                name="flatten"
            )

            x = self.flat_layer(x)

            feature_shape = x.shape

        feature_dim = feature_shape[-1]

        # Chỉ tạo classifier nếu backbone chưa phải classifier
        if feature_dim != self.num_classes:

            self.classifier_head = Dense(
                self.num_classes,
                activation=(
                    "sigmoid"
                    if self.num_classes == 2
                    else "softmax"
                ),
                name="classifier_head"
            )

            self.classifier_head.build(feature_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):

        x = self.backbone(
            inputs,
            training=training and self.backbone.trainable
        )

        if isinstance(x, (list, tuple)):
            x = x[-1]

        if self.custom_head is not None:
            x = self.custom_head(
                x,
                training=training and self.custom_head.trainable
            )

        if self.flat_layer is not None:
            x = self.flat_layer(x)

        if self.classifier_head is not None:
            x = self.classifier_head(x)

        return x

    @tf.function
    def predict(self, inputs):
        return self(inputs, training=False)

    def calc_loss(
        self,
        y_true,
        y_pred,
        loss_object,
        sample_weight=None
    ):
        loss = loss_object["loss"]
        loss.coefficient = loss_object["coeff"]

        return loss(
            y_true,
            y_pred,
            sample_weight=sample_weight
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "backbone":
                tf.keras.utils.serialize_keras_object(
                    self.backbone
                ),

            "custom_head":
                (
                    tf.keras.utils.serialize_keras_object(
                        self.custom_head
                    )
                    if self.custom_head is not None
                    else None
                ),

            "num_classes": self.num_classes,
            "name": self.name,
        })

        return config

    @classmethod
    def from_config(cls, config):

        backbone = tf.keras.utils.deserialize_keras_object(
            config.pop("backbone")
        )

        custom_head_cfg = config.pop("custom_head")

        custom_head = (
            tf.keras.utils.deserialize_keras_object(
                custom_head_cfg
            )
            if custom_head_cfg is not None
            else None
        )

        return cls(
            backbone=backbone,
            custom_head=custom_head,
            **config
        )

    def get_build_config(self):

        return {
            "input_shape": self._saved_input_shape
        }

    def build_from_config(self, config):

        self.build(config["input_shape"])