import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

from utils.post_processing import get_labels
from utils.logger import logger


class TrainModel(tf.keras.Model):
    def __init__(
        self, 
        architecture,
        classes=None,
        image_size=(224, 224, 3),
        global_clipnorm=5.,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        self.classes = classes
        self.image_size = image_size
        self.global_clipnorm = global_clipnorm
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.model_param_call = {}

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_object = loss
        self.list_metrics = metrics or []

    @property
    def metrics(self):
        return [self.total_loss_tracker] + self.list_metrics

    def _compute_loss(self, images, labels, training):
        self.model_param_call["training"] = training
        y_pred = self.architecture(images, **self.model_param_call)
        loss_value = sum(
            self.architecture.calc_loss(y_true=labels, y_pred=y_pred, loss_object=loss)
            for loss in self.loss_object
        )
        loss_value += tf.reduce_sum(self.architecture.losses)
        return y_pred, loss_value
        
    def train_step(self, data):
        images, labels = data
        
        with tf.GradientTape() as tape:
            y_pred, loss_value = self._compute_loss(images, labels, training=True)
            
        gradients = tape.gradient(loss_value, self.architecture.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.global_clipnorm)
        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.total_loss_tracker.update_state(loss_value)
        
        for metric in self.list_metrics:
            metric.update_state(labels, y_pred)
            
        return {m.name: m.result() for m in self.metrics} | {"learning_rate": self.optimizer.learning_rate}

    def test_step(self, data):
        images, labels = data
        y_pred, loss_value = self._compute_loss(images, labels, training=False)
        self.total_loss_tracker.update_state(loss_value)
        
        for metric in self.list_metrics:
            metric.update_state(labels, y_pred)
            
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        try:
            return self.predict(inputs)
        except Exception as e:
            logger.warning(f"Error in call(): {e}")
            return inputs

    @tf.function
    def predict(self, inputs):
        return self.architecture(inputs, training=False)

    def save_weights(self, weight_path, save_head=True, **kwargs):
        try:
            if save_head:
                self.architecture.save_weights(weight_path, **kwargs)
                logger.info(f"Full model weights saved to: {weight_path}")
            else:
                backbone_path = weight_path.replace(".weights.h5", "_backbone.weights.h5")
                self.architecture.backbone.save_weights(backbone_path, **kwargs)
                logger.info(f"Backbone weights saved to: {backbone_path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

    def load_weights(self, weight_path):
        try:
            self.architecture.build(input_shape=self.image_size)
            self.architecture.built = True
            self.architecture.load_weights(weight_path, skip_mismatch=True)
            logger.info(f"Weights loaded from: {weight_path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")

    def save_model(self, model_path, save_head=True):
        try:
            if save_head:
                self.architecture.save(model_path)
                logger.info(f"Full model saved to: {model_path}")
            else:
                backbone_path = model_path.replace(".keras", "_backbone.keras")
                self.architecture.backbone.save(backbone_path)
                logger.info(f"Backbone saved to: {backbone_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, model_path):
        try:
            self.architecture = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
