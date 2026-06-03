import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Adapters")
class DenseLoRA(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_config,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        train_base=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_config = dict(layer_config)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.train_base = bool(train_base)

        base_config = dict(self.layer_config)
        self.activation_name = base_config.get("activation", "linear")
        base_config["activation"] = "linear"
        self.base_layer = tf.keras.layers.Dense.from_config(base_config)
        self.base_layer.trainable = self.train_base
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.activation = tf.keras.activations.get(self.activation_name)

    def build(self, input_shape):
        self.base_layer.build(input_shape)
        input_dim = int(input_shape[-1])
        units = int(self.layer_config["units"])

        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(input_dim, self.rank),
            initializer="random_normal",
            trainable=True,
        )
        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank, units),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        base = self.base_layer(inputs)
        dropped = self.dropout_layer(inputs, training=training)
        lora = tf.linalg.matmul(dropped, self.lora_a)
        lora = tf.linalg.matmul(lora, self.lora_b)
        output = base + lora * (self.alpha / self.rank)
        return self.activation(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "layer_config": self.layer_config,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "train_base": self.train_base,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="Adapters")
class Conv2DLoRA(tf.keras.layers.Layer):
    def __init__(
        self,
        layer_config,
        rank=8,
        alpha=16.0,
        dropout=0.0,
        train_base=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_config = dict(layer_config)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.train_base = bool(train_base)

        base_config = dict(self.layer_config)
        self.activation_name = base_config.get("activation", "linear")
        base_config["activation"] = "linear"
        self.base_layer = tf.keras.layers.Conv2D.from_config(base_config)
        self.base_layer.trainable = self.train_base
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.activation = tf.keras.activations.get(self.activation_name)

        self.lora_a = tf.keras.layers.Conv2D(
            filters=self.rank,
            kernel_size=self.layer_config["kernel_size"],
            strides=self.layer_config.get("strides", (1, 1)),
            padding=self.layer_config.get("padding", "valid"),
            dilation_rate=self.layer_config.get("dilation_rate", (1, 1)),
            use_bias=False,
            kernel_initializer="random_normal",
            name="lora_a",
        )
        self.lora_b = tf.keras.layers.Conv2D(
            filters=self.layer_config["filters"],
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="zeros",
            name="lora_b",
        )

    def build(self, input_shape):
        self.base_layer.build(input_shape)
        self.lora_a.build(input_shape)
        self.lora_b.build(self.lora_a.compute_output_shape(input_shape))
        super().build(input_shape)

    def call(self, inputs, training=False):
        base = self.base_layer(inputs)
        dropped = self.dropout_layer(inputs, training=training)
        lora = self.lora_a(dropped)
        lora = self.lora_b(lora)
        output = base + lora * (self.alpha / self.rank)
        return self.activation(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "layer_config": self.layer_config,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "train_base": self.train_base,
        })
        return config


def is_lora_layer(layer):
    return isinstance(layer, (DenseLoRA, Conv2DLoRA))


def _normalize_targets(target_layers):
    if not target_layers:
        return {"Dense", "Conv2D"}
    return set(target_layers)


def _should_wrap_layer(layer, target_layers, target_names):
    if target_names and layer.name not in target_names:
        return False

    if isinstance(layer, tf.keras.layers.Dense):
        return "Dense" in target_layers

    if isinstance(layer, tf.keras.layers.Conv2D):
        groups = getattr(layer, "groups", 1)
        return "Conv2D" in target_layers and groups == 1

    return False


def _make_lora_layer(layer, rank, alpha, dropout, train_base):
    if isinstance(layer, tf.keras.layers.Dense):
        return DenseLoRA(
            layer.get_config(),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            train_base=train_base,
            name=layer.name,
        )

    if isinstance(layer, tf.keras.layers.Conv2D):
        return Conv2DLoRA(
            layer.get_config(),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            train_base=train_base,
            name=layer.name,
        )

    return layer


def _copy_lora_weights(source_model, target_model):
    for source_layer in source_model.layers:
        try:
            target_layer = target_model.get_layer(source_layer.name)
        except ValueError:
            continue

        if is_lora_layer(target_layer):
            target_layer.base_layer.set_weights(source_layer.get_weights())
        elif hasattr(source_layer, "layers") and hasattr(target_layer, "layers"):
            _copy_lora_weights(source_layer, target_layer)
        elif source_layer.weights and target_layer.weights:
            target_layer.set_weights(source_layer.get_weights())


def set_lora_trainable(layer, trainable_base=False):
    if is_lora_layer(layer):
        layer.trainable = True
        layer.base_layer.trainable = trainable_base
        layer.lora_a.trainable = True
        layer.lora_b.trainable = True
        return

    children = getattr(layer, "layers", [])
    if children:
        layer.trainable = True
    else:
        layer.trainable = False

    for child in children:
        set_lora_trainable(child, trainable_base=trainable_base)


def _enable_head_trainable(layer):
    if is_lora_layer(layer):
        layer.trainable = True
        layer.lora_a.trainable = True
        layer.lora_b.trainable = True
        return

    layer.trainable = True
    for child in getattr(layer, "layers", []):
        _enable_head_trainable(child)


def _iter_layers_recursive(layer):
    for child in getattr(layer, "layers", []):
        yield child
        yield from _iter_layers_recursive(child)


def apply_lora(
    model,
    enabled=True,
    rank=8,
    alpha=16.0,
    dropout=0.0,
    target_layers=None,
    target_names=None,
    train_base=False,
    train_head=True,
    **kwargs
):
    if not enabled:
        return model

    del kwargs
    target_layers = _normalize_targets(target_layers)
    target_names = set(target_names or [])

    def clone_function(layer):
        if _should_wrap_layer(layer, target_layers, target_names):
            return _make_lora_layer(
                layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                train_base=train_base,
            )
        return layer.__class__.from_config(layer.get_config())

    lora_model = tf.keras.models.clone_model(
        model,
        clone_function=clone_function,
        recursive=True,
    )
    if model.built:
        lora_model.build(model.input_shape)
    _copy_lora_weights(model, lora_model)
    set_lora_trainable(lora_model, trainable_base=train_base)

    if train_head:
        for layer in _iter_layers_recursive(lora_model):
            if layer.name == "classifier_head":
                _enable_head_trainable(layer)

    return lora_model
