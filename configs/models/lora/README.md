# LoRA Configs

Supported keys under `Model.LoRA`:

- `enabled`: turns LoRA injection on or off.
- `type`: currently only `"lora"` is implemented.
- `rank`: low-rank adapter size.
- `alpha`: LoRA scale. Effective scale is `alpha / rank`.
- `dropout`: dropout before the LoRA branch.
- `target_layers`: layer classes to wrap. Supported: `"Conv2D"`, `"Dense"`.
- `target_names`: optional exact layer names to wrap. Use `null` to wrap all matching target layer classes.
- `train_base`: keep original wrapped layer weights trainable. Default LoRA behavior is `False`.
- `train_head`: keep a layer named `classifier_head` trainable when present.

Use `lora-conv2d-dense.yaml` for CNN classification backbones such as RepVGG.
Use `lora-dense-only.yaml` for ViT/MLP-style backbones where most trainable projections are `Dense` layers.
