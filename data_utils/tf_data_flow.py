import os
import cv2
import numpy as np
import tensorflow as tf
from random import shuffle

from augmenter import build_augmenter
from data_utils import Augmentor, Normalizer
from utils.auxiliary_processing import change_color_space
from utils.post_processing import resize_image, preprocess_input


class TFDataPipeline:
    def __init__(self,
                 dataset,
                 target_size,
                 batch_size,
                 color_space='RGB',
                 augmentor=None,
                 normalizer='divide',
                 mean_norm=None,
                 std_norm=None,
                 interpolation="BILINEAR",
                 phase='train',
                 num_workers=1,
                 debug_mode=False):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.target_size = target_size
        self.color_space = color_space
        self.phase       = phase
        self.debug_mode  = debug_mode
        self.num_workers = num_workers
        self.N           = len(self.dataset)

        if phase == "train":
            shuffle(self.dataset)

        self.augmentor = augmentor.get(phase) if isinstance(augmentor, dict) else augmentor
        if augmentor and isinstance(self.augmentor, (tuple, list)):
            self.augmentor = Augmentor(augment_objects=build_augmenter(self.augmentor))
            
        self.normalizer = Normalizer(normalizer,
                                     target_size=target_size,
                                     mean=mean_norm,
                                     std=std_norm,
                                     interpolation=interpolation)

    def load_data(self, sample):
        sample_image = sample.get('image')
        sample_label = sample['label']
        deep_channel = 1 if (len(self.target_size) > 2 and self.target_size[-1] > 1) else 0

        if sample_image is not None:
            image = sample_image
        else:
            img_path = os.path.join(sample['path'], sample['filename'])
            cv_imread_flag = cv2.IMREAD_COLOR if deep_channel else cv2.IMREAD_GRAYSCALE
            image = cv2.imread(img_path, cv_imread_flag)
            
        if self.color_space.lower() != 'bgr':
            image = change_color_space(image, 'bgr' if deep_channel else 'gray', self.color_space)

        if self.augmentor:
            image = self.augmentor(image)
            
        image = self.normalizer(image)
        return image, sample_label

    def data_generator(self):
        # batch_count = 0
        batch_images = []
        batch_labels = []

        for idx, sample in enumerate(self.dataset):
            image, label = self.load_data(sample)
            batch_images.append(image)
            batch_labels.append(label)

            if len(batch_images) == self.batch_size or idx == self.N - 1:
                # batch_count += 1
                current_batch_size = len(batch_images)
                
                # tf.print(f"Batch {batch_count}: {current_batch_size} samples")
                yield np.array(batch_images), np.array(batch_labels)
                batch_images = []
                batch_labels = []

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.target_size), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if self.phase == 'train':
            dataset = dataset.repeat()
        
        return dataset


    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))