import tensorflow as tf
from utils.logger import logger



class TrainModel(tf.keras.Model):
    def __init__(
        self,
        architecture,
        classes=None,
        inputs=(224, 224, 3),
        teacher_models=None,
        distillation_type="",                     # [None, "base", "online", "self", "feature", "free"]
        temperature=3,
        alpha=0.1,
        model_clip_gradient=5.,
        gradient_accumulation_steps=1,
        sam_rho=0.,
        use_ema=False,
        compile_jit=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        self.teacher_models = teacher_models
        self.distillation_type = distillation_type
        self.classes = classes
        self.inputs = inputs
        self.temperature = temperature
        self.alpha = alpha
        self.model_clip_gradient = model_clip_gradient
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.sam_rho = sam_rho
        self.use_ema = use_ema
        self.compile_jit = compile_jit

        if teacher_models or distillation_type:
            self.distillation_loss_fn = tf.keras.losses.KLDivergence()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")        
        self.accumulated_gradients = None
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.model_param_call = {}
        self.list_metrics = []
        self.prev_logits = None
        
    def compile(self, optimizer, loss, teacher_optimizer=None, metrics=None, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_objects = loss
        self.teacher_optimizer = teacher_optimizer
        self.list_metrics = metrics or []
        
        for _ in range(3):
            dummy = self.architecture(tf.random.normal([1, 224, 224, 3]))
            
        self.model_variables = self.architecture.trainable_variables

        self.train_grad_fn = self.grad_caculator(phase="train")
        self.test_grad_fn = self.grad_caculator(phase="test")
        self.predict_fn = self.build_predict_step()
        
        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.ema.apply(self.model_variables)
            
    @property
    def metrics(self):
        return [self.total_loss_tracker] + self.list_metrics
        
    def reset_metrics(self):
        super().reset_metrics()
        self.step_counter.assign(0)
        if self.accumulated_gradients is not None:
            for accum_grad in self.accumulated_gradients:
                accum_grad.assign(tf.zeros_like(accum_grad))

    def _get_teacher_predictions(self, images, labels):
        preds_list = []
        if self.teacher_models:
            if self.distillation_type.lower() == "base":
                preds_list = [model(images, training=False) for model in self.teacher_models]
            elif self.distillation_type.lower() == "self":
                if hasattr(self, "prev_logits") and self.prev_logits is not None:
                    if tf.reduce_all(tf.equal(tf.shape(self.prev_logits), tf.shape(labels))):
                        preds_list.append(tf.stop_gradient(self.prev_logits))
                if hasattr(self, "ema"):
                    preds_list.append(self.ema(images, training=False))
            elif self.distillation_type.lower() == "online":
                preds_list = [model(images, training=True) for model in self.teacher_models]
        elif self.distillation_type.lower() == "self" and hasattr(self, "ema"):
            preds_list.append(self.ema(images, training=False))
        return preds_list
        
    def _compute_loss(self, images, labels, training, sample_weight=None):
        self.model_param_call["training"] = training
        logits = self.architecture(images, **self.model_param_call)

        if hasattr(self.architecture, "calc_loss"):
            loss_value = sum(
                self.architecture.calc_loss(loss_object=loss_object, y_true=labels, y_pred=logits, sample_weight=sample_weight)
                for loss_object in self.loss_objects
            )
        else:
            loss_value = sum(
                loss_object["loss"](y_true=labels, y_pred=logits, sample_weight=sample_weight) * loss_object["coeff"]
                for loss_object in self.loss_objects
            )

        return {
            "loss": loss_value,
            "logits": logits
        }

    def _apply_gradients(self, gradients):
        if self.gradient_accumulation_steps == 1:
            if self.model_clip_gradient > 0:
                gradients, _ = tf.clip_by_global_norm(gradients, self.model_clip_gradient)
    
            self.optimizer.apply_gradients(zip(gradients, self.model_variables))

        else:
            if self.accumulated_gradients is None:
                self.accumulated_gradients = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.model_variables]
    
            for accum, grad in zip(self.accumulated_gradients, gradients):
                if grad is not None:
                    accum.assign_add(grad)
    
            self.step_counter.assign_add(1)
            if self.step_counter % self.gradient_accumulation_steps == 0:
                if self.model_clip_gradient > 0:
                    clipped, _ = tf.clip_by_global_norm(self.accumulated_gradients, self.model_clip_gradient)
                else:
                    clipped = self.accumulated_gradients
    
                self.optimizer.apply_gradients(zip(clipped, self.model_variables))
                for accum in self.accumulated_gradients:
                    accum.assign(tf.zeros_like(accum))

    def _apply_gradients_online_teachers(self, images, student_logits):
        with tf.GradientTape() as tape:
            teacher_preds = [model(images, training=True) for model in self.teacher_models]
            teacher_logits = tf.reduce_mean(tf.stack(teacher_preds, axis=0), axis=0)
    
            teacher_distill_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature),
                tf.nn.softmax(student_logits / self.temperature)
            ) * (self.temperature ** 2)
    
        for model in self.teacher_models:
            grads = tape.gradient(teacher_distill_loss, model.trainable_variables)
            self.teacher_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def _apply_sam_gradients(
        self,
        tape,
        scaled_loss,
        images,
        labels,
        sample_weight,
    ):
        if self.sam_rho <= 0.:
            return tape.gradient(scaled_loss, self.model_variables)
    
        # First grads
        grads = tape.gradient(scaled_loss, self.model_variables)
        grad_norm = tf.linalg.global_norm(grads)
        scale = self.sam_rho / (grad_norm + 1e-12)
        e_ws = [(g * scale if g is not None else tf.zeros_like(v)) for g, v in zip(grads, self.model_variables)]
    
        # Perturb weights
        for v, e_w in zip(self.model_variables, e_ws):
            v.assign_add(e_w)
    
        # Second forward
        with tf.GradientTape() as advance_tape:
            caculated_loss = self._compute_loss(images, labels, training=True, sample_weight=sample_weight)
            advance_loss = caculated_loss["loss"]
            scaled_advance_loss = advance_loss / self.gradient_accumulation_steps
    
        advance_grads = advance_tape.gradient(scaled_advance_loss, self.model_variables)
    
        # Restore weights
        for v, e_w in zip(self.model_variables, e_ws):
            v.assign_sub(e_w)
    
        return [g if g is not None else tf.zeros_like(v) for g, v in zip(advance_grads, self.model_variables)]

    def grad_caculator(self, phase="test"):
        
        def _train_grad(images, labels, sample_weight=None):
            teacher_predictions_list = self._get_teacher_predictions(images, labels)
            
            with tf.GradientTape() as tape:
                caculated_loss = self._compute_loss(images, labels, training=True, sample_weight=sample_weight)
                loss_value = caculated_loss["loss"]
                logits = caculated_loss["logits"]
                
                if teacher_predictions_list:
                    teacher_logits = tf.reduce_mean(
                        tf.stack(teacher_predictions_list, axis=0),
                        axis=0,
                    )
                    
                    distillation_loss = self.distillation_loss_fn(
                        tf.nn.softmax(teacher_logits / self.temperature),
                        tf.nn.softmax(logits / self.temperature),
                    ) * (self.temperature ** 2)
                    
                    loss_value = self.alpha * loss_value + (1 - self.alpha) * distillation_loss
                
                scale_loss_value = loss_value / self.gradient_accumulation_steps
    
                final_grads = self._apply_sam_gradients(tape, scale_loss_value, images, labels, sample_weight)
    
            return {
                "loss": scale_loss_value,
                "grads": final_grads,
                "model_preds": logits,
                "teacher_preds": teacher_predictions_list
            }
            
        def _test_grad(images, labels, sample_weight=None):
            caculated_loss = self._compute_loss(images, labels, training=False, sample_weight=sample_weight)
            loss_value = caculated_loss["loss"]
            logits = caculated_loss["logits"]

            return {
                "loss": loss_value,
                "model_preds": logits,
            }

        if self.compile_jit:
            print(f"Use JIT compile in phase {phase.lower()}")
            
        return tf.function(_train_grad if phase.lower() == "train" else _test_grad, jit_compile=self.compile_jit)

    def train_step(self, data):
        images, labels, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        results = self.train_grad_fn(images, labels, sample_weight)
        self._apply_gradients(results["grads"])

        if (
            results["teacher_preds"] and
            self.teacher_models and 
            self.distillation_type.lower() == "online" and 
            hasattr(self, "teacher_optimizer")
        ):
            self._apply_gradients_online_teachers(images, tf.stop_gradient(results["model_preds"]))

        self.prev_logits = tf.stop_gradient(results["model_preds"])
                
        if hasattr(self, "ema"):
            self.ema.apply(self.model_variables)
            
        self.total_loss_tracker.update_state(results["loss"])

        for metric in self.list_metrics:
            metric.update_state(labels, results["model_preds"], sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        results = self.test_grad_fn(images, labels, sample_weight)
        
        self.total_loss_tracker.update_state(results["loss"])

        for metric in self.list_metrics:
            metric.update_state(labels, results["model_preds"], sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def build_predict_step(self):
        def _predict_step(data):
            if isinstance(data, dict):
                images, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
            else:
                images = data
            return self.architecture(images, training=False)
            
        return tf.function(_predict_step, jit_compile=self.compile_jit)
            
    def call(self, inputs, training=False):
        try:
            if self.use_ema:
                self.ema.apply(self.model_variables)

            return self.predict_fn(inputs)
        except Exception as e:
            logger.warning(f"Error in call(): {e}")
            return inputs

    def predict(self, inputs):
        return self.predict_fn(inputs)

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
            