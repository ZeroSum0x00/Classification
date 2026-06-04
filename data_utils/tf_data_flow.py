import os
import numpy as np
import tensorflow as tf
import cv2
from augmenter import build_augmenter
from data_utils import AugmentWrapper, Normalizer
from utils.auxiliary_processing import change_color_space


class TFDataPipeline:
    """tf.data based pipeline that accepts the same init args as
    `DataSequencePipeline` and preserves augmentor/normalizer behaviour.
    """
    def __init__(
        self,
        dataset,
        target_size,
        batch_size,
        color_space="RGB",
        augmentor=None,
        normalizer="divide",
        mean_norm=None,
        std_norm=None,
        sampler=None,
        interpolation="BILINEAR",
        phase="train",
        num_workers=1,
        debug_mode=False,
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.color_space = color_space
        self.interpolation = interpolation
        self.phase = phase
        self.debug_mode = debug_mode
        self.num_workers = num_workers
        self.N = len(self.dataset)

        # shuffle is handled at tf.data level for training

        # augmentor handling (same logic as in DataSequencePipeline)
        self.augmentor = augmentor.get(phase) if isinstance(augmentor, dict) else augmentor
        if augmentor and isinstance(self.augmentor, (tuple, list)):
            self.augmentor = AugmentWrapper(transforms=build_augmenter(self.augmentor))

        # normalizer handling (same as DataSequencePipeline)
        if isinstance(normalizer, dict):
            self.normalizer = Normalizer(
                normalizer.get("mode", "divide"),
                target_size=target_size,
                mean=normalizer.get("mean", None),
                std=normalizer.get("std", None),
                interpolation=normalizer.get("interpolation", "BILINEAR"),
                backend="tf",
            )
        elif isinstance(normalizer, str):
            self.normalizer = Normalizer(
                normalizer,
                target_size=target_size,
                mean=mean_norm,
                std=std_norm,
                backend="tf",
            )
        else:
            self.normalizer = normalizer
            if isinstance(self.normalizer, Normalizer) and self.normalizer.backend != "tf":
                self.normalizer = Normalizer(
                    norm_type=self.normalizer.norm_type,
                    target_size=self.normalizer.target_size,
                    mean=self.normalizer.mean,
                    std=self.normalizer.std,
                    interpolation=self.normalizer.interpolation,
                    backend="tf",
                )

        # class weights if sampler indicates balanced sampling
        if sampler and sampler.lower() in ["balance", "balanced"]:
            all_labels = [sample["label"] for sample in self.dataset]
            classes, counts = np.unique(all_labels, return_counts=True)
            # mimic sklearn's compute_class_weight behaviour roughly
            class_weights = {}
            total = len(all_labels)
            for i, cls in enumerate(classes):
                # weight = total / (n_classes * count)
                class_weights[int(cls)] = float(total) / (len(classes) * counts[i])
            self.class_weights = class_weights
        else:
            self.class_weights = None
            self.class_weight_table = None
        if self.class_weights:
            keys = tf.constant(list(self.class_weights.keys()), dtype=tf.int32)
            vals = tf.constant(list(self.class_weights.values()), dtype=tf.float32)
            init = tf.lookup.KeyValueTensorInitializer(keys, vals)
            self.class_weight_table = tf.lookup.StaticHashTable(init, default_value=1.0)

        # prepare data source: either all in-memory images or file paths
        self._all_in_memory = all(sample.get("image") is not None for sample in self.dataset)
        if self._all_in_memory:
            self._images_np = np.stack([sample.get("image") for sample in self.dataset], axis=0)
            self._labels_np = np.array([int(sample.get("label")) for sample in self.dataset], dtype=np.int32)
        else:
            self._paths = [os.path.join(s.get("path", ""), s.get("filename", "")) for s in self.dataset]
            self._labels_np = np.array([int(sample.get("label")) for sample in self.dataset], dtype=np.int32)

    def _process_index_numpy(self, idx):
        """Numpy-side processing called from `tf.py_function` using an index.
        Returns: image (float32, HWC), label (int32), sample_weight (float32)
        """
        # kept for compatibility if needed elsewhere; not used in new pipeline
        raise RuntimeError("_process_index_numpy is deprecated; TFDataPipeline uses tf.data mappings now")

    def _tf_change_color(self, image, current_space, to_space):
        # image: tf.Tensor HWC, channels last, uint8 or float
        cur = current_space.lower()
        to = to_space.lower()
        img = image
        if cur == to:
            return img
        # normalize to float in [0,1] for conversions where needed
        if img.dtype != tf.float32:
            img_f = tf.image.convert_image_dtype(img, tf.float32)
        else:
            img_f = img

        if cur == "bgr" and to == "rgb":
            img_out = tf.reverse(img, axis=[-1])
            return img_out
        if cur == "rgb" and to == "bgr":
            return tf.reverse(img, axis=[-1])
        if cur == "rgb" and to == "hsv":
            return tf.image.rgb_to_hsv(img_f)
        if cur == "hsv" and to == "rgb":
            return tf.image.hsv_to_rgb(img_f)
        if to == "gray":
            if cur == "rgb":
                gray = tf.image.rgb_to_grayscale(img_f)
            elif cur == "bgr":
                # reverse then rgb_to_grayscale
                gray = tf.image.rgb_to_grayscale(tf.reverse(img, axis=[-1]))
            else:
                gray = tf.image.rgb_to_grayscale(img_f)
            return gray
        # fallback: return as-is (may still be float)
        return img_f

    def _tf_resize(self, img_tf):
        if self.target_size is None or len(self.target_size) < 2:
            return img_tf

        method = tf.image.ResizeMethod.BILINEAR
        if isinstance(self.interpolation, str) and self.interpolation.upper() == "NEAREST":
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

        img_tf = tf.image.resize(img_tf, self.target_size[:2], method=method)
        return img_tf

    def _tf_normalize(self, img_tf):
        if self.normalizer is None:
            if img_tf.dtype != tf.float32:
                return tf.image.convert_image_dtype(img_tf, tf.float32)
            return img_tf
        return self.normalizer.normalize(img_tf)

    def _augment_numpy(self, img_np, lbl_np):
        # called inside tf.py_function: img_np may be a numpy array or a Tensor
        # returns numpy array (uint8 or float32) after augmentation
        if not self.augmentor:
            return img_np
        if hasattr(img_np, "numpy"):
            img_np = img_np.numpy()
        img_np = np.asarray(img_np)
        metadata = {"image": img_np, "label": int(lbl_np)}
        out = self.augmentor(metadata)
        aug_img = out.get("image") if isinstance(out, dict) else out
        return aug_img

    def get_dataset(self):
        if self._all_in_memory:
            ds = tf.data.Dataset.from_tensor_slices((self._images_np, self._labels_np))
            if self.phase == "train":
                ds = ds.shuffle(buffer_size=self.N).repeat()

            def _map_mem(img, lbl):
                # ensure uint8
                img = tf.convert_to_tensor(img)
                if img.dtype != tf.uint8:
                    img = tf.image.convert_image_dtype(img, tf.uint8)
                # call augmentor via py_function if exists
                if self.augmentor:
                    aug = tf.py_function(func=lambda im, l: self._augment_numpy(im, l), inp=[img, lbl], Tout=tf.uint8)
                    aug.set_shape([None, None, int(self.target_size[-1])])
                    img_tf = tf.image.convert_image_dtype(aug, tf.float32)
                else:
                    img_tf = tf.image.convert_image_dtype(img, tf.float32)

                img_tf = self._tf_resize(img_tf)
                img_tf = self._tf_normalize(img_tf)
                if self.class_weight_table is not None:
                    sw_tf = self.class_weight_table.lookup(tf.cast(lbl, tf.int32))
                else:
                    sw_tf = tf.constant(1.0, dtype=tf.float32)
                return img_tf, lbl, sw_tf

            ds = ds.map(_map_mem, num_parallel_calls=self.num_workers if self.num_workers > 0 else tf.data.AUTOTUNE)

        else:
            ds = tf.data.Dataset.from_tensor_slices((self._paths, self._labels_np))
            if self.phase == "train":
                ds = ds.shuffle(buffer_size=self.N).repeat()

            def _map_path(path, lbl):
                img_bytes = tf.io.read_file(path)
                channels = self.target_size[-1]
                img = tf.image.decode_image(img_bytes, channels=channels)
                # ensure shape known rank for resize
                img.set_shape([None, None, int(channels)])
                method = tf.image.ResizeMethod.BILINEAR
                if isinstance(self.interpolation, str) and self.interpolation.upper() == "NEAREST":
                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                img = tf.image.resize(img, self.target_size[:2], method=method)
                img = tf.cast(img, tf.uint8)

                # convert color spaces if needed: TF decode gives RGB by default
                img_conv = self._tf_change_color(img, current_space="rgb", to_space=self.color_space)

                if self.augmentor:
                    aug = tf.py_function(func=lambda im, l: self._augment_numpy(im, l), inp=[img_conv, lbl], Tout=tf.uint8)
                    aug.set_shape([None, None, int(self.target_size[-1])])
                    img_tf = tf.image.convert_image_dtype(aug, tf.float32)
                else:
                    img_tf = tf.image.convert_image_dtype(img_conv, tf.float32)

                img_tf = self._tf_resize(img_tf)
                img_tf = self._tf_normalize(img_tf)
                if self.class_weight_table is not None:
                    sw_tf = self.class_weight_table.lookup(tf.cast(lbl, tf.int32))
                else:
                    sw_tf = tf.constant(1.0, dtype=tf.float32)
                return img_tf, lbl, sw_tf

            ds = ds.map(_map_path, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        # allow non-deterministic ordering for speed if desired
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)

        return ds
