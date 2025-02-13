import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer
from augmenter import build_augmenter
from data_utils.data_augmentation import Augmentor
from utils.auxiliary_processing import random_range, change_color_space
from utils.post_processing import get_labels
from utils.logger import logger


def get_train_test_data(data_dirs, 
                        classes         = None, 
                        target_size     = [224, 224, 3], 
                        batch_size      = 16, 
                        color_space     = 'RGB',
                        augmentor       = None,
                        normalizer      = None,
                        mean_norm       = None,
                        std_norm        = None,
                        data_type       = 'dirname',
                        check_data      = False, 
                        load_memory     = False,
                        dataloader_mode = 0,
                        *args, **kwargs):
    """
        dataloader_mode = 0:   train - validation - test
        dataloader_mode = 1:   train - validation
        dataloader_mode = 2:   train
    """

    if classes:
        if isinstance(classes, str):
            classes, _ = get_labels(classes)
    else:
        classes, _ = get_labels(data_dirs)

    data_train = get_data(data_dirs,
                          classes     = classes,
                          data_type   = data_type,
                          phase       = 'train', 
                          check_data  = check_data,
                          load_memory = load_memory)

    train_generator = Data_Sequence(data_train, 
                                    target_size = target_size, 
                                    batch_size  = batch_size, 
                                    color_space = color_space,
                                    augmentor   = augmentor,
                                    normalizer  = normalizer,
                                    mean_norm   = mean_norm,
                                    std_norm    = std_norm,
                                    phase       = "train",
                                    *args, **kwargs)
    if dataloader_mode != 2:
        data_valid = get_data(data_dirs,
                              classes     = classes,
                              data_type   = data_type,
                              phase       = 'validation', 
                              check_data  = check_data,
                              load_memory = load_memory)

        valid_generator = Data_Sequence(data_valid, 
                                        target_size = target_size, 
                                        batch_size  = batch_size, 
                                        color_space = color_space,
                                        augmentor   = augmentor,
                                        normalizer  = normalizer,
                                        mean_norm   = mean_norm,
                                        std_norm    = std_norm,
                                        phase       = "valid",
                                        *args, **kwargs)
    else:
        valid_generator = None
        
    if dataloader_mode == 0:
        data_test = get_data(data_dirs,
                             classes     = classes,
                             data_type   = data_type,
                             phase       = 'test', 
                             check_data  = check_data,
                             load_memory = load_memory)

        test_generator = Data_Sequence(data_test, 
                                       target_size = target_size, 
                                       batch_size  = batch_size, 
                                       color_space = color_space,
                                       augmentor   = augmentor,
                                       normalizer  = normalizer,
                                       mean_norm   = mean_norm,
                                       std_norm    = std_norm,
                                       phase       = "test",
                                       *args, **kwargs)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator


class Data_Sequence(Sequence):
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
                 debug_mode=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.target_size = target_size

        if phase == "train":
            self.dataset   = shuffle(self.dataset)
        self.N = self.n = len(self.dataset)
        
        self.color_space = color_space
        self.phase       = phase
        self.debug_mode  = debug_mode

        self.augmentor = augmentor.get(phase) if isinstance(augmentor, dict) else augmentor
        if augmentor and isinstance(self.augmentor, (tuple, list)):
            self.augmentor = Augmentor(augment_objects=build_augmenter(self.augmentor))

        self.normalizer = Normalizer(normalizer, mean=mean_norm, std=std_norm)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = []
        batch_label = []
        debug_image = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i           = i % self.N
            sample = self.dataset[i]
            label = sample['label']

            if len(self.target_size) == 2 or self.target_size[-1] == 1:
                deep_channel = 0
            else:
                deep_channel = 1

            if sample['image'] is not None:
                image = sample['image']
                image = change_color_space(image, 'bgr', self.color_space if deep_channel else 'gray')
            else:
                img_path = os.path.join(sample['path'], sample['filename'])
                image = cv2.imread(img_path, deep_channel)
                image = change_color_space(image, 'bgr' if deep_channel else 'gray', self.color_space)

            if self.augmentor:
                image = self.augmentor(image)

            image = self.normalizer(image,
                                    target_size=self.target_size,
                                    interpolation=cv2.INTER_NEAREST)

            batch_image.append(image)
            batch_label.append(label)
            
            if self.debug_mode:
                debug_image.append(img_path)

        batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)
        
        if self.debug_mode:
            return batch_image, batch_label, debug_image
        else:
            return batch_image, batch_label
    
    def on_epoch_end(self):
        if self.phase:
            self.dataset = shuffle(self.dataset)
