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
            
        self.normalizer = Normalizer(normalizer, mean=mean_norm, std=std_norm)

    def load_data(self, sample):
        sample_image = sample.get('image')
        sample_label = sample['label']

        if len(self.target_size) == 2 or self.target_size[-1] == 1:
            deep_channel = 0
        else:
            deep_channel = 1

        if sample_image:
            image = sample_image
        else:
            img_path = os.path.join(sample['path'], sample['filename'])
            cv_imread_flag = cv2.IMREAD_COLOR if deep_channel else cv2.IMREAD_GRAYSCALE
            image = cv2.imread(img_path, cv_imread_flag)
            
        if self.color_space.lower() != 'bgr':
            image = change_color_space(image, 'bgr' if deep_channel else 'gray', self.color_space)

        if self.augmentor:
            image = self.augmentor(image)
            
        image = self.normalizer(image,
                                target_size=self.target_size,
                                interpolation=cv2.INTER_NEAREST)
        return image, sample_label

    def data_generator(self):
        for sample in self.dataset:
            yield self.load_data(sample)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.target_size,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE).repeat()
        return dataset

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))