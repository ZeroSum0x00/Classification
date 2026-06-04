import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.train_model import TrainModel


class MeanModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer="ones",
        )

    def call(self, inputs, training=False):
        x = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=False)
        x = tf.expand_dims(x, axis=-1)
        return self.dense(x)


class SoftmaxModel(tf.keras.Model):
    def __init__(self, bias):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            2,
            activation="softmax",
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(bias),
        )

    def call(self, inputs, training=False):
        x = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=False)
        x = tf.expand_dims(x, axis=-1)
        return self.dense(x)


class MaxPoolModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.pool(inputs)
        x = self.flatten(x)
        return self.dense(x)


def test_apply_gradients_averages_accumulated_gradients_and_flushes_remainder():
    model = TrainModel(
        architecture=tf.keras.Sequential(),
        gradient_accumulation_steps=2,
        model_clip_gradient=0.0,
    )
    variable = tf.Variable(1.0)

    model.model_variables = [variable]
    model.optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    model.optimizer.build(model.model_variables)
    model.accumulated_gradients = [
        tf.Variable(tf.zeros_like(variable), trainable=False)
    ]
    model.steps_per_epoch = 3

    did_apply = model._apply_gradients([tf.constant(2.0)])
    assert did_apply.numpy() == False
    tf.debugging.assert_near(variable, 1.0)

    did_apply = model._apply_gradients([tf.constant(4.0)])
    assert did_apply.numpy() == True
    tf.debugging.assert_near(variable, -2.0)

    did_apply = model._apply_gradients([tf.constant(6.0)])
    assert did_apply.numpy() == True
    tf.debugging.assert_near(variable, -8.0)


def test_train_grad_returns_unscaled_loss_for_logging():
    model = TrainModel(
        architecture=MeanModel(),
        gradient_accumulation_steps=4,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.MeanSquaredError(), "coeff": 1.0}],
    )

    images = tf.ones([1, 224, 224, 3])
    labels = tf.zeros([1, 1])
    results = model.train_grad_fn(images, labels)

    tf.debugging.assert_near(results["loss"], 1.0)


def test_sam_gradient_restores_perturbed_weights():
    model = TrainModel(
        architecture=MeanModel(),
        gradient_accumulation_steps=1,
        sam_rho=0.1,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.MeanSquaredError(), "coeff": 1.0}],
    )

    before = [tf.identity(v) for v in model.model_variables]
    images = tf.ones([1, 224, 224, 3])
    labels = tf.zeros([1, 1])

    results = model.train_grad_fn(images, labels)

    for actual, expected in zip(model.model_variables, before):
        tf.debugging.assert_near(actual, expected)
    assert all(grad is not None for grad in results["grads"])


def test_distillation_loss_is_zero_for_matching_predictions():
    model = TrainModel(
        architecture=MeanModel(),
        distillation_type="base",
        temperature=2.0,
    )

    predictions = tf.constant([[0.8, 0.2]], dtype=tf.float32)
    loss = model._distillation_loss(predictions, predictions)

    tf.debugging.assert_near(loss, 0.0)


def test_base_distillation_does_not_update_teacher_weights():
    teacher = SoftmaxModel([2.0, -2.0])
    model = TrainModel(
        architecture=SoftmaxModel([0.0, 0.0]),
        teacher_models=[teacher],
        distillation_type="base",
        alpha=0.0,
        temperature=2.0,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.CategoricalCrossentropy(), "coeff": 1.0}],
    )

    before = [tf.identity(v) for v in teacher.trainable_variables]
    images = tf.ones([1, 224, 224, 3])
    labels = tf.constant([[1.0, 0.0]])

    model.train_step((images, labels))

    for actual, expected in zip(teacher.trainable_variables, before):
        tf.debugging.assert_near(actual, expected)


def test_self_distillation_uses_ema_teacher_without_leaving_swapped_weights():
    model = TrainModel(
        architecture=SoftmaxModel([0.0, 0.0]),
        distillation_type="self",
        use_ema=True,
        alpha=0.5,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.CategoricalCrossentropy(), "coeff": 1.0}],
    )

    before = [tf.identity(v) for v in model.model_variables]
    images = tf.ones([1, 224, 224, 3])
    labels = tf.constant([[1.0, 0.0]])

    results = model.train_grad_fn(images, labels)

    assert len(results["teacher_preds"]) == 1
    for actual, expected in zip(model.model_variables, before):
        tf.debugging.assert_near(actual, expected)


def test_predict_uses_ema_weights_when_available():
    model = TrainModel(
        architecture=MeanModel(),
        use_ema=True,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.MeanSquaredError(), "coeff": 1.0}],
    )

    images = tf.ones([1, 224, 224, 3])
    model.model_variables[0].assign([[3.0]])
    model.ema_variables[0].assign([[1.0]])

    prediction = model.predict(images)

    tf.debugging.assert_near(prediction, [[1.0]])
    tf.debugging.assert_near(model.model_variables[0], [[3.0]])


def test_compile_jit_true_forces_jit_for_max_pooling_layers():
    model = TrainModel(
        architecture=MaxPoolModel(),
        inputs=(8, 8, 1),
        compile_jit=True,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.MeanSquaredError(), "coeff": 1.0}],
    )

    assert model.compile_jit is True


def test_compile_jit_auto_is_disabled_for_max_pooling_layers():
    model = TrainModel(
        architecture=MaxPoolModel(),
        inputs=(8, 8, 1),
        compile_jit="auto",
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.MeanSquaredError(), "coeff": 1.0}],
    )

    assert model.compile_jit is False


def test_online_distillation_can_update_teacher_weights():
    teacher = SoftmaxModel([-2.0, 2.0])
    model = TrainModel(
        architecture=SoftmaxModel([2.0, -2.0]),
        teacher_models=[teacher],
        distillation_type="online",
        alpha=0.5,
        temperature=2.0,
        model_clip_gradient=0.0,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        teacher_optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=[{"loss": tf.keras.losses.CategoricalCrossentropy(), "coeff": 1.0}],
    )

    before = [tf.identity(v) for v in teacher.trainable_variables]
    images = tf.ones([1, 224, 224, 3])
    labels = tf.constant([[1.0, 0.0]])

    model.train_step((images, labels))

    changed = [
        tf.reduce_any(tf.not_equal(actual, expected))
        for actual, expected in zip(teacher.trainable_variables, before)
    ]
    assert any(bool(item.numpy()) for item in changed)
