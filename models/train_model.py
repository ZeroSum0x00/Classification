import os

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
        model_summary=False,
        save_model_format="weights",
        save_model_head=True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.architecture = architecture
        self.teacher_models = teacher_models
        self.distillation_type = (distillation_type or "").lower()
        self.classes = classes
        self.inputs = inputs
        self.temperature = max(float(temperature or 1.0), 1e-6)
        self.alpha = float(alpha)
        self.model_clip_gradient = float(model_clip_gradient or 0.0)
        self.gradient_accumulation_steps = max(
            1,
            int(gradient_accumulation_steps or 1),
        )
        self.sam_rho = float(sam_rho or 0.0)
        self.use_ema = use_ema
        self.compile_jit_mode = self._normalize_compile_jit_mode(compile_jit)
        self.compile_jit = self.compile_jit_mode in {"auto", "true"}
        self.model_summary = model_summary
        self.save_model_format = self._normalize_save_format(save_model_format)
        self.save_model_head = save_model_head

        if teacher_models or self.distillation_type:
            self.distillation_loss_fn = tf.keras.losses.KLDivergence()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accumulated_gradients = None
        self.accumulated_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.steps_per_epoch = None
        self.model_param_call = {}
        self.list_metrics = []
        self.ema_decay = 0.99

    def _normalize_save_format(self, save_format):
        value = str(save_format or "weights").lower()
        if value in {"weight", "weights"}:
            return "weights"
        if value == "model":
            return "model"
        raise ValueError(
            "save_format must be one of 'weight', 'weights', or 'model'. "
            f"Got: {save_format}"
        )

    def _normalize_compile_jit_mode(self, compile_jit):
        if isinstance(compile_jit, str):
            value = compile_jit.lower()
            if value == "auto":
                return "auto"
            if value in {"1", "true", "yes", "y", "on"}:
                return "true"
            if value in {"0", "false", "no", "n", "off"}:
                return "false"
            raise ValueError(
                "compile_jit must be one of True, False, or 'auto'. "
                f"Got: {compile_jit}"
            )
        return "true" if bool(compile_jit) else "false"

    def _has_jit_unsafe_layer(self, layer):
        layer_name = layer.__class__.__name__.lower()
        if "maxpool" in layer_name or "max_pool" in layer_name:
            return True

        return any(
            self._has_jit_unsafe_layer(child)
            for child in getattr(layer, "layers", [])
        )

    def _resolve_compile_jit(self):
        if self.compile_jit_mode == "false":
            return False

        if (
            self.compile_jit_mode == "auto" and
            self._has_jit_unsafe_layer(self.architecture)
        ):
            logger.warning(
                "compile_jit='auto' disabled JIT because the model contains "
                "MaxPooling/MaxPool layers, which can fail under XLA for some "
                "TensorFlow device backends. Set compile_jit=True to force JIT."
            )
            return False

        return True
        
    def compile(self, optimizer, loss, teacher_optimizer=None, metrics=None, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss_objects = loss
        self.teacher_optimizer = teacher_optimizer
        self.list_metrics = metrics or []
        
        dummy = tf.random.normal([1, *self.inputs])
        for _ in range(3):
            self.architecture(dummy)

        if self.model_summary:
            self.architecture.summary(expand_nested=True)
            
        self.model_variables = self.architecture.trainable_variables

        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model_variables)
            
        if self.gradient_accumulation_steps > 1:
            self.accumulated_gradients = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.model_variables
            ]
        
        if self.use_ema:
            self.ema_variables = [
                tf.Variable(v, trainable=False)
                for v in self.model_variables
            ]

        if self.teacher_models:
            for model in self.teacher_models:
                model(dummy, training=False)

            if self.teacher_optimizer is not None and hasattr(self.teacher_optimizer, "build"):
                teacher_variables = [
                    var
                    for model in self.teacher_models
                    for var in model.trainable_variables
                ]
                self.teacher_optimizer.build(teacher_variables)

        self.compile_jit = self._resolve_compile_jit()

        self.train_grad_fn = self.grad_caculator(phase="train")
        self.test_grad_fn = self.grad_caculator(phase="test")
        self.predict_fn = self.build_predict_step()
            
    @property
    def metrics(self):
        return [self.total_loss_tracker] + self.list_metrics

    def fit(self, *args, **kwargs):
        self.steps_per_epoch = kwargs.get("steps_per_epoch", None)
        return super().fit(*args, **kwargs)
        
    def reset_metrics(self):
        did_flush = self._flush_accumulated_gradients()
        self._apply_ema_if_needed(did_flush)
        super().reset_metrics()
        self.step_counter.assign(0)
        self.accumulated_steps.assign(0)
        if self.accumulated_gradients is not None:
            for accum_grad in self.accumulated_gradients:
                accum_grad.assign(tf.zeros_like(accum_grad))

    def _get_teacher_predictions(self, images, labels):
        preds_list = []
        if self.teacher_models:
            if self.distillation_type == "base":
                preds_list = [
                    tf.stop_gradient(model(images, training=False))
                    for model in self.teacher_models
                ]
            elif self.distillation_type == "online":
                preds_list = [
                    tf.stop_gradient(model(images, training=True))
                    for model in self.teacher_models
                ]

        if self.distillation_type == "self" and hasattr(self, "ema_variables"):
            preds_list.append(self._call_with_ema_variables(images))
        return preds_list
        
    def _compute_loss(self, images, labels, training, sample_weight=None):
        if not training and hasattr(self, "ema_variables"):
            logits = self._call_with_ema_variables(images)
        else:
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

    def _normalize_gradients(self, gradients):
        return [
            grad if grad is not None else tf.zeros_like(var)
            for grad, var in zip(gradients, self.model_variables)
        ]

    def _clip_gradients(self, gradients):
        if self.model_clip_gradient > 0:
            gradients, _ = tf.clip_by_global_norm(
                gradients,
                self.model_clip_gradient,
            )
        return gradients

    def _apply_ema(self):
        if hasattr(self, "ema_variables"):
            for ema_var, var in zip(self.ema_variables, self.model_variables):
                ema_var.assign(
                    self.ema_decay * ema_var +
                    (1.0 - self.ema_decay) * var
                )
        return tf.constant(True)

    def _apply_ema_if_needed(self, did_apply_gradients):
        if hasattr(self, "ema_variables"):
            return tf.cond(
                did_apply_gradients,
                self._apply_ema,
                lambda: tf.constant(False),
            )
        return tf.constant(False)

    def _apply_accumulated_gradients(self):
        denominator = tf.maximum(self.accumulated_steps, 1)
        gradients = [
            accum / tf.cast(denominator, accum.dtype)
            for accum in self.accumulated_gradients
        ]
        gradients = self._clip_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model_variables))

        for accum in self.accumulated_gradients:
            accum.assign(tf.zeros_like(accum))
        self.accumulated_steps.assign(0)
        return tf.constant(True)

    def _flush_accumulated_gradients(self):
        if self.gradient_accumulation_steps == 1 or self.accumulated_gradients is None:
            return tf.constant(False)

        return tf.cond(
            self.accumulated_steps > 0,
            self._apply_accumulated_gradients,
            lambda: tf.constant(False),
        )

    def _is_last_epoch_step(self):
        if self.steps_per_epoch is None:
            return tf.constant(False)
        return self.step_counter >= self.steps_per_epoch

    def _apply_gradients(self, gradients):
        gradients = self._normalize_gradients(gradients)
        self.step_counter.assign_add(1)

        if self.gradient_accumulation_steps == 1:
            gradients = self._clip_gradients(gradients)
            self.optimizer.apply_gradients(zip(gradients, self.model_variables))
            return tf.constant(True)

        for accum, grad in zip(self.accumulated_gradients, gradients):
            accum.assign_add(grad)
        self.accumulated_steps.assign_add(1)

        should_apply = tf.logical_or(
            self.accumulated_steps >= self.gradient_accumulation_steps,
            self._is_last_epoch_step(),
        )

        return tf.cond(
            should_apply,
            self._apply_accumulated_gradients,
            lambda: tf.constant(False),
        )

    def _call_with_ema_variables(self, images):
        current_values = [
            tf.identity(var)
            for var in self.model_variables
        ]

        for var, ema_var in zip(self.model_variables, self.ema_variables):
            var.assign(ema_var)

        predictions = self.architecture(images, training=False)

        for var, value in zip(self.model_variables, current_values):
            var.assign(value)

        return tf.stop_gradient(predictions)

    def _to_distillation_distribution(self, predictions):
        predictions = tf.cast(predictions, tf.float32)
        eps = tf.keras.backend.epsilon()

        if predictions.shape.rank is not None and predictions.shape[-1] == 1:
            probs = tf.clip_by_value(predictions, eps, 1.0 - eps)
            probs = tf.concat([1.0 - probs, probs], axis=-1)
        else:
            probs = tf.clip_by_value(predictions, eps, 1.0)
            probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)

        log_probs = tf.math.log(probs) / self.temperature
        return tf.nn.softmax(log_probs, axis=-1)

    def _distillation_loss(self, teacher_predictions, student_predictions):
        teacher_probs = self._to_distillation_distribution(
            tf.stop_gradient(teacher_predictions)
        )
        student_probs = self._to_distillation_distribution(student_predictions)

        return self.distillation_loss_fn(
            teacher_probs,
            student_probs,
        ) * (self.temperature ** 2)

    def _combine_with_distillation(self, loss_value, logits, teacher_predictions_list):
        if teacher_predictions_list:
            teacher_logits = tf.reduce_mean(
                tf.stack(teacher_predictions_list, axis=0),
                axis=0,
            )
            distillation_loss = self._distillation_loss(teacher_logits, logits)
            
            return self.alpha * loss_value + (1 - self.alpha) * distillation_loss
        return loss_value

    def _apply_gradients_online_teachers(self, images, student_logits):
        if self.teacher_optimizer is None:
            return

        student_logits = tf.stop_gradient(student_logits)
        for model in self.teacher_models:
            with tf.GradientTape() as tape:
                teacher_logits = model(images, training=True)
                teacher_distill_loss = self._distillation_loss(
                    student_logits,
                    teacher_logits,
                )

            grads = tape.gradient(
                teacher_distill_loss,
                model.trainable_variables,
            )
            self.teacher_optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def _apply_sam_gradients(
        self,
        tape,
        loss_value,
        images,
        labels,
        sample_weight,
        teacher_predictions_list,
    ):
        if self.sam_rho <= 0.:
            return tape.gradient(loss_value, self.model_variables)
    
        # First grads
        grads = self._normalize_gradients(
            tape.gradient(loss_value, self.model_variables)
        )
        grad_norm = tf.linalg.global_norm(grads)
        scale = self.sam_rho / (grad_norm + 1e-12)
        e_ws = [g * scale for g in grads]
    
        # Perturb weights
        for v, e_w in zip(self.model_variables, e_ws):
            v.assign_add(e_w)
    
        # Second forward
        with tf.GradientTape() as advance_tape:
            caculated_loss = self._compute_loss(images, labels, training=True, sample_weight=sample_weight)
            advance_loss = caculated_loss["loss"]
            advance_logits = caculated_loss["logits"]
            advance_loss = self._combine_with_distillation(
                advance_loss,
                advance_logits,
                teacher_predictions_list,
            )
    
        advance_grads = advance_tape.gradient(advance_loss, self.model_variables)
    
        # Restore weights
        for v, e_w in zip(self.model_variables, e_ws):
            v.assign_sub(e_w)
    
        return self._normalize_gradients(advance_grads)

    def grad_caculator(self, phase="test"):
        
        def _train_grad(images, labels, sample_weight=None):
            teacher_predictions_list = self._get_teacher_predictions(images, labels)
            
            with tf.GradientTape() as tape:
                caculated_loss = self._compute_loss(images, labels, training=True, sample_weight=sample_weight)
                loss_value = caculated_loss["loss"]
                logits = caculated_loss["logits"]
                loss_value = self._combine_with_distillation(
                    loss_value,
                    logits,
                    teacher_predictions_list,
                )
    
                final_grads = self._apply_sam_gradients(
                    tape,
                    loss_value,
                    images,
                    labels,
                    sample_weight,
                    teacher_predictions_list,
                )
    
            return {
                "loss": loss_value,
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
            logger.info(f"Use JIT compile in phase {phase.lower()}")
            
        return tf.function(_train_grad if phase.lower() == "train" else _test_grad, jit_compile=self.compile_jit)

    def train_step(self, data):
        images, labels, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        results = self.train_grad_fn(images, labels, sample_weight)
        did_apply_gradients = self._apply_gradients(results["grads"])

        if (
            results["teacher_preds"] and
            self.teacher_models and 
            self.distillation_type == "online" and 
            self.teacher_optimizer is not None
        ):
            self._apply_gradients_online_teachers(images, tf.stop_gradient(results["model_preds"]))

        self._apply_ema_if_needed(did_apply_gradients)
            
        self.total_loss_tracker.update_state(results["loss"])

        for metric in self.list_metrics:
            metric.update_state(labels, results["model_preds"], sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        did_flush = self._flush_accumulated_gradients()
        self._apply_ema_if_needed(did_flush)
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
            if hasattr(self, "ema_variables"):
                return self._call_with_ema_variables(images)
            return self.architecture(images, training=False)
            
        return tf.function(_predict_step, jit_compile=self.compile_jit)
            
    def call(self, inputs, training=False):
        try:
            return self.predict_fn(inputs)
        except Exception as e:
            logger.warning(f"Error in call(): {e}")
            return inputs

    def predict(self, inputs):
        return self.predict_fn(inputs)

    def checkpoint_path(self, directory, stem):
        ext = ".keras" if self.save_model_format == "model" else ".weights.h5"
        return os.path.join(directory, f"{stem}{ext}")

    def save(self, path, **kwargs):
        try:
            if self.save_model_format == "model":
                if self.save_model_head:
                    self.architecture.save(path)
                    logger.info(f"Full model saved to: {path}")
                else:
                    backbone_path = path.replace(".keras", "_backbone.keras")
                    self.architecture.backbone.save(backbone_path)
                    logger.info(f"Backbone saved to: {backbone_path}")
            else:
                if self.save_model_head:
                    self.architecture.save_weights(path, **kwargs)
                    logger.info(f"Full model weights saved to: {path}")
                else:
                    backbone_path = path.replace(".weights.h5", "_backbone.weights.h5")
                    self.architecture.backbone.save_weights(backbone_path, **kwargs)
                    logger.info(f"Backbone weights saved to: {backbone_path}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")

    def load(self, path):
        try:
            if self.save_model_format == "model":
                self.architecture = tf.keras.models.load_model(path)
                logger.info(f"Model loaded from: {path}")
            else:
                self.architecture.build(input_shape=self.inputs)
                self.architecture.built = True
                self.architecture.load_weights(path, skip_mismatch=True)
                logger.info(f"Weights loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            
