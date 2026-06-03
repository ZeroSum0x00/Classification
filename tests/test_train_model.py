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
