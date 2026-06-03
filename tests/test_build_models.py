import sys
from pathlib import Path

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models


def make_trainer_config(train_mode="scratch", freeze_layer=-1):
    return {
        "strategy": {
            "train_mode": train_mode,
            "freeze_layer": freeze_layer,
        },
        "advanced": {
            "model_clip_gradient": 5.0,
            "gradient_accumulation_steps": 1,
            "sam_rho": 0.0,
            "train_with_ema": False,
            "compile_jit": False,
        },
    }


def make_model_config(weight_path=None, custom_head=None, lora=None):
    config = {
        "inputs": [16, 16, 3],
        "classes": ["cat", "dog", "bird"],
        "is_set_classes": True,
        "weight_path": weight_path,
        "Architecture": {
            "name": "TinyCLS",
        },
        "Backbone": {
            "name": "TinyBackbone",
        },
    }

    if custom_head is not None:
        config["CustomHead"] = custom_head

    if lora is not None:
        config["LoRA"] = lora

    return config


def tiny_backbone(inputs=(16, 16, 3), num_classes=3, include_head=True, **kwargs):
    image = tf.keras.Input(shape=inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv")(image)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)

    if include_head:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", name="dense"),
                tf.keras.layers.Dense(num_classes, activation="softmax", name="logits"),
            ],
            name="classifier_head",
        )(x)

    return tf.keras.Model(image, x, name="tiny_backbone")


class TinyCLS(tf.keras.Model):
    def __init__(self, backbone, custom_head=None, num_classes=3, name="TinyCLS", **kwargs):
        super().__init__(name=name, **kwargs)
        self.backbone = backbone
        self.custom_head = custom_head
        self.num_classes = num_classes
        self.classifier_head = None

    def build(self, input_shape):
        if not self.backbone.built:
            self.backbone.build(input_shape)

        dummy = tf.zeros([1, *input_shape[1:]])
        x = self.backbone(dummy, training=False)

        if self.custom_head is not None:
            x = self.custom_head(x, training=False)

        if x.shape[-1] != self.num_classes:
            self.classifier_head = tf.keras.layers.Dense(
                self.num_classes,
                activation="softmax",
                name="classifier_head",
            )
            self.classifier_head.build(x.shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training and self.backbone.trainable)

        if self.custom_head is not None:
            x = self.custom_head(x, training=training and self.custom_head.trainable)

        if self.classifier_head is not None:
            x = self.classifier_head(x)

        return x


@pytest.fixture(autouse=True)
def patch_dynamic_import(monkeypatch):
    def fake_dynamic_import(name, globals_dict):
        if name == "TinyBackbone":
            return tiny_backbone
        if name == "TinyCLS":
            return TinyCLS
        raise ValueError(f"Unexpected import: {name}")

    monkeypatch.setattr(models, "dynamic_import", fake_dynamic_import)


def test_build_models_scratch_creates_train_model():
    model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(),
    )

    assert model.architecture.num_classes == 3
    assert model.architecture.backbone.name == "tiny_backbone"


def test_build_models_feature_extractor_freezes_backbone():
    model = models.build_models(
        make_trainer_config("feature-extractor"),
        make_model_config(),
    )

    full_transfer_model = model.architecture.backbone

    assert full_transfer_model.get_layer("conv").trainable is False
    assert full_transfer_model.get_layer("pool").trainable is False
    assert full_transfer_model.output_shape[-1] == 8


def test_build_models_feature_extraction_alias_is_supported():
    model = models.build_models(
        make_trainer_config("feature-extraction"),
        make_model_config(),
    )

    assert model.architecture.backbone.name == "full_transfer_model"


def test_build_models_fine_tuning_uses_configured_freeze_layer():
    model = models.build_models(
        make_trainer_config("fine-tuning", freeze_layer=0),
        make_model_config(),
    )

    assert model.architecture.backbone.get_layer("conv").trainable is False
    assert model.architecture.backbone.get_layer("pool").trainable is False


def test_build_models_loads_h5_weights_for_scratch(tmp_path):
    base_model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(),
    ).architecture
    base_model.build((None, 16, 16, 3))

    weight_path = tmp_path / "tiny.weights.h5"
    base_model.save_weights(weight_path)

    model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(weight_path=str(weight_path)),
    )

    assert model.architecture.built is True


def test_build_models_rejects_unknown_weight_format():
    with pytest.raises(ValueError, match="Unsupported weight_path format"):
        models.build_models(
            make_trainer_config("scratch"),
            make_model_config(weight_path="model.ckpt"),
        )


def test_build_models_applies_lora_to_conv2d_and_dense():
    model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(
            lora={
                "enabled": True,
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_layers": ["Conv2D", "Dense"],
            }
        ),
    )

    backbone = model.architecture.backbone
    conv = backbone.get_layer("conv")
    dense = backbone.get_layer("classifier_head").get_layer("dense")

    assert conv.__class__.__name__ == "Conv2DLoRA"
    assert dense.__class__.__name__ == "DenseLoRA"
    assert conv.base_layer.trainable is False
    assert dense.base_layer.trainable is False
    assert conv.lora_a.trainable is True
    assert dense.lora_a.trainable is True


def test_build_models_loads_h5_weights_before_applying_lora(tmp_path):
    base_model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(),
    ).architecture
    base_model.build((None, 16, 16, 3))

    images = tf.random.uniform([2, 16, 16, 3], seed=1)
    expected = base_model(images, training=False)

    weight_path = tmp_path / "tiny.weights.h5"
    base_model.save_weights(weight_path)

    model = models.build_models(
        make_trainer_config("scratch"),
        make_model_config(
            weight_path=str(weight_path),
            lora={
                "enabled": True,
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_layers": ["Conv2D", "Dense"],
            },
        ),
    )

    actual = model.architecture(images, training=False)

    assert model.architecture.backbone.get_layer("conv").__class__.__name__ == "Conv2DLoRA"
    tf.debugging.assert_near(actual, expected, atol=1e-6)
