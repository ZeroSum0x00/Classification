import tensorflow as tf
from utils.logger import logger


class TrainModel(tf.keras.Model):
    def __init__(
        self,
        architecture,
        classes=None,
        inputs=(224, 224, 3),
        use_ema=False,
        model_clip_gradient=5.,
        gradient_accumulation_steps=1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        self.classes = classes
        self.inputs = inputs
        self.model_clip_gradient = model_clip_gradient
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")        
        self.accumulated_gradients = None
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.model_param_call = {}
        self.list_metrics = []
        
        if use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.ema.apply(self.architecture.trainable_variables)
            
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_object = loss
        self.list_metrics = metrics or []

    @property
    def metrics(self):
        return [self.total_loss_tracker] + self.list_metrics

    def _compute_loss(self, images, labels, training, sample_weight=None):
        self.model_param_call["training"] = training
        y_pred = self.architecture(images, **self.model_param_call)
        loss_value = sum(
            self.architecture.calc_loss(y_true=labels, y_pred=y_pred, loss_object=loss, sample_weight=sample_weight)
            for loss in self.loss_object
        )
        loss_value += tf.reduce_sum(self.architecture.losses)
        return y_pred, loss_value
        
    def reset_metrics(self):
        super().reset_metrics()
        self.step_counter.assign(0)
        if self.accumulated_gradients is not None:
            for accum_grad in self.accumulated_gradients:
                accum_grad.assign(tf.zeros_like(accum_grad))

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            images, labels, sample_weight = data
        else:
            images, labels = data
            sample_weight = None

        with tf.GradientTape() as tape:
            y_pred, loss_value = self._compute_loss(images, labels, training=True, sample_weight=sample_weight)
            scale_loss_value = loss_value / tf.constant(self.gradient_accumulation_steps, dtype=tf.float32)

        gradients = tape.gradient(scale_loss_value, self.architecture.trainable_variables)
        
        if self.gradient_accumulation_steps == 1:
            if self.model_clip_gradient > 0:
                gradients, _ = tf.clip_by_global_norm(gradients, self.model_clip_gradient)
            self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        else:
            if self.accumulated_gradients is None:
                self.accumulated_gradients = [
                    tf.Variable(tf.zeros_like(var), trainable=False)
                    for var in self.architecture.trainable_variables
                ]
                
            for accum_grad, grad in zip(self.accumulated_gradients, gradients):
                if grad is not None:
                    accum_grad.assign_add(grad)
    
            self.step_counter.assign_add(1)
            if self.step_counter % self.gradient_accumulation_steps == 0:
                if self.model_clip_gradient > 0:
                    clipped_grads, _ = tf.clip_by_global_norm(self.accumulated_gradients, self.model_clip_gradient)
                else:
                    clipped_grads = self.accumulated_gradients
            
                self.optimizer.apply_gradients(zip(clipped_grads, self.architecture.trainable_variables))
            
                for accum_grad in self.accumulated_gradients:
                    accum_grad.assign(tf.zeros_like(accum_grad))

        if hasattr(self, "ema"):
            self.ema.apply(self.architecture.trainable_variables)
            
        self.total_loss_tracker.update_state(loss_value)
    
        for metric in self.list_metrics:
            metric.update_state(labels, y_pred, sample_weight=sample_weight)
    
        return {m.name: m.result() for m in self.metrics}
        
    @tf.function
    def test_step(self, data):
        if len(data) == 3:
            images, labels, sample_weight = data
        else:
            images, labels = data
            sample_weight = None
        
        y_pred, loss_value = self._compute_loss(images, labels, training=False, sample_weight=sample_weight)
        self.total_loss_tracker.update_state(loss_value)

        for metric in self.list_metrics:
            metric.update_state(labels, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        try:
            if self.use_ema:
                self.ema.apply(self.architecture.trainable_variables)

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
            self.architecture.build(input_shape=self.inputs)
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
            