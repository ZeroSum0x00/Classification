import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.post_processing import get_labels
from utils.logger import logger


class TrainModel(tf.keras.Model):
    def __init__(self, 
                 architecture,
                 classes=None,
                 **kwargs):
        super(TrainModel, self).__init__()
        self.architecture = architecture
        self.classes      = classes
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(TrainModel, self).compile()
        self.optimizer = optimizer
        self.loss_object = loss
        self.list_metrics = metrics

    @property
    def metrics(self):
        if self.list_metrics:
            return [
                self.total_loss_tracker,
                *self.list_metrics,
            ]
        else:
            return [self.total_loss_tracker]

    def train_step(self, data):
        images, labels = data
        loss_value = 0

        with tf.GradientTape() as tape:
            for losses in self.loss_object:
                y_pred = self.architecture(images, training=True)
                loss_value += self.architecture.calc_loss(y_true=labels, 
                                                          y_pred=y_pred, 
                                                          loss_object=losses)

            loss_value  = tf.reduce_sum(self.architecture.losses) + loss_value

        gradients = tape.gradient(loss_value, self.architecture.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.total_loss_tracker.update_state(loss_value)

        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(labels, y_pred)

        results = {}
        for metric in self.metrics:
            if isinstance(metric.result(), dict):
                for k, v in metric.result().items():
                    results[k] = v
            else:
                results[metric.name] = metric.result()

        results['learning_rate'] = self.optimizer.learning_rate
        return results
    
    def test_step(self, data):
        images, labels = data
        loss_value = 0

        for losses in self.loss_object:
            y_pred = self.architecture(images, training=False)
            loss_value += self.architecture.calc_loss(y_true=labels, 
                                                      y_pred=y_pred, 
                                                      loss_object=losses)

        loss_value  = tf.reduce_sum(self.architecture.losses) + loss_value

        self.total_loss_tracker.update_state(loss_value)
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(labels, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        try:
            pred = self.predict(inputs)
            return pred
        except:
            return inputs

    @tf.function
    def predict(self, inputs):
        pred = self.architecture(inputs, training=False)
        return pred

    def save_weights(self, weight_path, save_head=True, save_format='tf', **kwargs):
        if save_head:
            self.architecture.save_weights(weight_path, **kwargs)

    # def load_weights(self, weight_objects):
    #     for weight in weight_objects:
    #         weight_path = weight['path']
    #         custom_objects = weight['custom_objects']
    #         if weight_path:
    #             self.architecture.build(input_shape=self.image_size)
    #             self.architecture.built = True
    #             self.architecture.load_weights(weight_path)
    #             logger.info("Load Classification weights from {}".format(weight_path))

    # def save_models(self, weight_path, save_format='tf'):
    #     self.architecture.save(weight_path, save_format=save_format)

    # def load_models(self, weight_objects):
    #     for weight in weight_objects:
    #         weight_path = weight['path']
    #         custom_objects = weight['custom_objects']
    #         if weight_path:
    #             self.architecture = load_model(weight_path, custom_objects=custom_objects)
    #             logger.info("Load STR model from {}".format(weight_path))

    def get_config(self):
        config = super().get_config()
        config.update({
                "architecture": self.architecture,
                "total_loss_tracker": self.total_loss_tracker,
                "optimizer": self.optimizer
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)