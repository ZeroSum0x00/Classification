import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from multiprocessing.pool import ThreadPool

from augmenter import build_augmenter
from data_utils import Augmentor, Normalizer
from utils.auxiliary_processing import change_color_space
from utils.post_processing import resize_image, preprocess_input


class DataSequencePipeline(Sequence):
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
                 debug_mode=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.target_size = target_size
        self.color_space = color_space
        self.phase       = phase
        self.debug_mode  = debug_mode
        self.num_workers = num_workers
        
        if phase == "train":
            shuffle(self.dataset)
        self.N = len(self.dataset)
        

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

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError(f"Index {index} out of range!")

        batch_indices = np.arange(index * self.batch_size, min((index + 1) * self.batch_size, self.N))

        if self.num_workers > 1:
            with ThreadPool(self.num_workers) as pool:
                batch_data = pool.map(self.load_data, [self.dataset[i % self.N] for i in batch_indices])
        else:
            batch_data = [self.load_data(self.dataset[i % self.N]) for i in batch_indices]

        batch_image, batch_label = zip(*batch_data)
        batch_image = np.stack(batch_image)
        batch_label = np.array(batch_label)
        
        if self.debug_mode:
            return batch_image, batch_label, [self.dataset[i]['path'] for i in batch_indices]
        else:
            return batch_image, batch_label
    
    def on_epoch_end(self):
        if self.phase:
            shuffle(self.dataset)