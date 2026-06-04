import numpy as np
import pytest
import tensorflow as tf

from models.architectures.vgg.repvgg import (
    RepVGGBlock,
    RepVGG_A0,
    repvgg_reparameter,
)


def _max_output_diff(model_a, model_b, input_shape, seed=0):
    tf.random.set_seed(seed)
    inputs = tf.random.normal([4, *input_shape])
    out_a = model_a(inputs, training=False)
    out_b = model_b(inputs, training=False)
    return float(tf.reduce_max(tf.abs(out_a - out_b)))


def test_repvgg_block_convert_without_identity_branch():
  """Stem/downsample blocks have no identity branch."""
  block = RepVGGBlock(filters=32, strides=(2, 2), deploy=False, name="stem")
  inputs = tf.random.normal([2, 16, 16, 3])
  block(inputs, training=False)

  kernel, bias = block.repvgg_convert()
  assert kernel.shape == (3, 3, 3, 32)
  assert bias.shape == (32,)


def test_repvgg_block_convert_with_identity_branch():
  block = RepVGGBlock(filters=32, strides=(1, 1), deploy=False, name="inner")
  inputs = tf.random.normal([2, 16, 16, 32])
  block(inputs, training=False)

  kernel, bias = block.repvgg_convert()
  assert kernel.shape == (3, 3, 32, 32)
  assert bias.shape == (32,)


def test_repvgg_block_train_matches_deploy_after_fusion():
  train_block = RepVGGBlock(filters=48, strides=(1, 1), deploy=False, name="fusion")
  deploy_block = RepVGGBlock(filters=48, strides=(1, 1), deploy=True, name="fusion_deploy")

  inputs = tf.random.normal([2, 14, 14, 48])
  train_block(inputs, training=False)

  kernel, bias = train_block.repvgg_convert()
  deploy_block(inputs, training=False)
  deploy_block.rbr_reparam.layers[1].set_weights([kernel, bias])

  train_out = train_block(inputs, training=False)
  deploy_out = deploy_block(inputs, training=False)
  max_diff = float(tf.reduce_max(tf.abs(train_out - deploy_out)))

  assert max_diff < 1e-4, f"train/deploy block mismatch: {max_diff}"


def test_repvgg_a0_reparameter_matches_training_outputs():
  input_shape = [32, 32, 3]
  num_classes = 10

  train_model = RepVGG_A0(
    inputs=input_shape,
    include_head=True,
    weights=None,
    num_classes=num_classes,
    deploy=False,
  )
  deploy_model = RepVGG_A0(
    inputs=input_shape,
    include_head=True,
    weights=None,
    num_classes=num_classes,
    deploy=True,
  )

  train_model(tf.random.normal([2, *input_shape]), training=False)
  repvgg_reparameter(
    train_model,
    deploy_model,
    input_shape=input_shape,
    classes=num_classes,
  )

  max_diff = _max_output_diff(train_model, deploy_model, input_shape)
  assert max_diff < 1e-4, f"full model reparam mismatch: {max_diff}"


def test_repvgg_a0_deploy_has_fewer_train_branches():
  deploy_model = RepVGG_A0(
    inputs=[32, 32, 3],
    include_head=True,
    weights=None,
    num_classes=10,
    deploy=True,
  )
  deploy_model(tf.random.normal([1, 32, 32, 3]), training=False)

  repvgg_layers = [
    layer
    for layer in deploy_model.layers
    if isinstance(layer, RepVGGBlock)
  ]
  assert repvgg_layers, "expected RepVGGBlock layers in deploy model"
  assert all(hasattr(layer, "rbr_reparam") for layer in repvgg_layers)
  assert all(not hasattr(layer, "rbr_dense") for layer in repvgg_layers)


@pytest.mark.parametrize("model_fn", [RepVGG_A0])
def test_repvgg_reparameter_is_idempotent_on_deploy_model(model_fn):
  input_shape = [32, 32, 3]
  num_classes = 5

  train_model = model_fn(
    inputs=input_shape,
    include_head=True,
    weights=None,
    num_classes=num_classes,
    deploy=False,
  )
  deploy_model = model_fn(
    inputs=input_shape,
    include_head=True,
    weights=None,
    num_classes=num_classes,
    deploy=True,
  )

  train_model(tf.random.normal([2, *input_shape]), training=False)
  repvgg_reparameter(train_model, deploy_model, input_shape=input_shape, classes=num_classes)

  before = deploy_model.get_weights()
  repvgg_reparameter(train_model, deploy_model, input_shape=input_shape, classes=num_classes)
  after = deploy_model.get_weights()

  for w1, w2 in zip(before, after):
    np.testing.assert_allclose(w1, w2, rtol=0, atol=0)
