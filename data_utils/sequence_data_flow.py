import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence
from multiprocessing.pool import ThreadPool

from augmenter import build_augmenter
from data_utils import AugmentWrapper, Normalizer
from utils.auxiliary_processing import change_color_space



class DataSequencePipeline(Sequence):
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
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.color_space = color_space
        self.interpolation = interpolation
        self.phase = phase
        self.debug_mode = debug_mode
        self.num_workers = num_workers
        self.N = len(self.dataset)

        if phase == "train":
            self.dataset = shuffle(self.dataset)

        self.augmentor = augmentor.get(phase) if isinstance(augmentor, dict) else augmentor
        if augmentor and isinstance(self.augmentor, (tuple, list)):
            self.augmentor = AugmentWrapper(transforms=build_augmenter(self.augmentor))

        if isinstance(normalizer, str):
            self.normalizer = Normalizer(
                normalizer,
                target_size=target_size,
                mean=mean_norm,
                std=std_norm
            )
        else:
            self.normalizer = normalizer

        if sampler and sampler.lower() in ["balance", "balanced"]:
            all_labels = [sample['label'] for sample in self.dataset]
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(all_labels),
                y=all_labels,
            )
            self.class_weights = dict(enumerate(class_weights))
        else:
            self.class_weights = None
            
    def load_data(self, sample):
        sample_image = sample.get("image")
        sample_label = sample["label"]
        metadata = {
            "image": None,
            "label": -1,
        }

        deep_channel = 1 if (len(self.target_size) > 2 and self.target_size[-1] > 1) else 0

        if sample_image is not None:
            image = sample_image
        else:
            img_path = os.path.join(sample["path"], sample["filename"])
            cv_imread_flag = cv2.IMREAD_COLOR if deep_channel else cv2.IMREAD_GRAYSCALE
            image = cv2.imread(img_path, cv_imread_flag)

        if self.color_space.lower() != "bgr":
            image = change_color_space(image, "bgr" if deep_channel else "gray", self.color_space)

        metadata["image"] = image
        metadata["label"] = sample_label

        if self.augmentor:
            metadata = self.augmentor(metadata)

        metadata = self.normalizer(metadata)
        return metadata
 
    def collate_batch(self, batch_list):
        def merge_key(key_items):
            first = key_items[0]
            if isinstance(first, dict):
                return {
                    k: merge_key([item.get(k) for item in key_items])
                    for k in first
                }
            elif isinstance(first, np.ndarray):
                try:
                    return np.stack(key_items, axis=0)
                except:
                    for l in key_items:
                        return np.array(key_items)
            elif isinstance(first, int):
                return np.array(key_items)
            else:
                return first

        return {
            k: merge_key([item[k] for item in batch_list])
            for k in batch_list[0]
        }

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError(f"Index {index} out of range!")

        batch_indices = np.arange(index * self.batch_size, min((index + 1) * self.batch_size, self.N))

        if self.num_workers > 1:
            with ThreadPool(self.num_workers) as pool:
                batch_data = pool.map(self.load_data, [self.dataset[i] for i in batch_indices])
        else:
            batch_data = [self.load_data(self.dataset[i]) for i in batch_indices]

        batch_data = self.collate_batch(batch_data)

        if self.class_weights:
            batch_weights = np.array([self.class_weights.get(label, 1.0) for label in batch_data["label"]], dtype=np.float32)
        else:
            batch_weights = np.ones_like(batch_data["label"], dtype=np.float32)

        images = batch_data["image"]
        labels = batch_data["label"]
        sample_weight = batch_weights
        return images, labels, sample_weight
            
    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def on_epoch_end(self):
        if self.phase == "train":
            self.dataset = shuffle(self.dataset)
            
