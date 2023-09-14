import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer
from data_utils.data_augmentation import Augmentor
from utils.auxiliary_processing import random_range, change_color_space
from utils.logger import logger


def get_train_test_data(data_zipfile, 
                        dst_dir,
                        classes, 
                        target_size, 
                        batch_size, 
                        init_epoch,
                        end_epoch,
                        color_space,
                        augmentor,
                        normalizer,
                        mean_norm,
                        std_norm,
                        data_type,
                        check_data, 
                        load_memory,
                        dataloader_mode=0,
                        *args, **kwargs):
    """
        dataloader_mode = 0:   train - validation - test
        dataloader_mode = 1:   train - validation
        dataloader_mode = 2:   train
    """
    data_folder = extract_data_folder(data_zipfile, dst_dir)
    data_train = get_data(data_folder,
                          classes           = classes,
                          data_type         = data_type,
                          phase             = 'train', 
                          check_data        = check_data,
                          load_memory       = load_memory)

    train_generator = Data_Sequence(data_train, 
                                    target_size             = target_size, 
                                    batch_size              = batch_size, 
                                    classes                 = classes,
                                    color_space             = color_space,
                                    augmentor               = augmentor,
                                    init_epoch              = init_epoch,
                                    end_epoch               = end_epoch,
                                    normalizer              = normalizer,
                                    mean_norm               = mean_norm,
                                    std_norm                = std_norm,
                                    phase                   = "train",
                                    *args, **kwargs)
    if dataloader_mode != 2:
        data_valid = get_data(data_folder,
                              classes           = classes,
                              data_type         = data_type,
                              phase             = 'validation', 
                              check_data        = check_data,
                              load_memory       = load_memory)
        valid_generator = Data_Sequence(data_valid, 
                                        target_size             = target_size, 
                                        batch_size              = batch_size, 
                                        classes                 = classes,
                                        color_space             = color_space,
                                        augmentor               = augmentor,
                                        init_epoch              = init_epoch,
                                        end_epoch               = end_epoch,
                                        normalizer              = normalizer,
                                        mean_norm               = mean_norm,
                                        std_norm                = std_norm,
                                        phase                   = "valid",
                                        *args, **kwargs)
    else:
        valid_generator = None
        
    if dataloader_mode == 1:
        data_test = get_data(data_folder,
                             classes           = classes,
                             data_type         = data_type,
                             phase             = 'test', 
                             check_data        = check_data,
                             load_memory       = load_memory)
        test_generator = Data_Sequence(data_valid, 
                                       target_size             = target_size, 
                                       batch_size              = batch_size, 
                                       classes                 = classes,
                                       color_space             = color_space,
                                       augmentor               = augmentor,
                                       init_epoch              = init_epoch,
                                       end_epoch               = end_epoch,
                                       normalizer              = normalizer,
                                       mean_norm               = mean_norm,
                                       std_norm                = std_norm,
                                       phase                   = "test",
                                       *args, **kwargs)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator


class Data_Sequence(Sequence):
    def __init__(self, dataset, target_size, 
                 batch_size, classes, color_space='RGB',
                 augmentor=None, init_epoch=0, 
                 end_epoch=100, normalizer=None, mean_norm=None, std_norm=None, phase='train', debug_mode=False):
        self.data_path = dataset['data_path']
        self.dataset   = dataset['data_extractor']
        self.batch_size = batch_size
        self.target_size = target_size
        
        self.classes = classes
        self.num_class = len(self.classes)

        if phase == "train":
            self.dataset   = shuffle(self.dataset)
        self.N = self.n = len(self.dataset)
        
        self.color_space = color_space
        self.current_epoch = init_epoch
        self.end_epoch   = end_epoch
        self.phase       = phase
        self.debug_mode  = debug_mode
        
        if augmentor and isinstance(augmentor[phase], list):
            self.augmentor = Augmentor(augment_objects=augmentor[phase], 
                                       target_size=target_size)
        else:
            self.augmentor = augmentor[phase]

        self.normalizer = Normalizer(normalizer, mean=mean_norm, std=std_norm)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = []
        batch_label = []
        debug_label = []
        debug_image = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N
            sample = self.dataset[i]
            label_name = sample['label']
            img_path = f"{self.data_path}{label_name}/{sample['filename']}"
            image = cv2.imread(img_path, 1)
            image = change_color_space(image, 'bgr', self.color_space)
            if self.augmentor:
                image = self.augmentor(image)
            image = self.normalizer(image,
                                    target_size=self.target_size,
                                    interpolation=cv2.INTER_NEAREST)

            label_index = self.classes.index(label_name)
            if self.num_class == 2:
                batch_label.append(label_index)
            else:
                label = np.zeros(self.num_class, dtype=np.float32)
                label[label_index] = 1
                batch_label.append(label)
            batch_image.append(image)
            
            if self.debug_mode:
                debug_image.append(img_path)
                debug_label.append(label_name)
                
        batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)
        
        if self.debug_mode:
            return batch_image, batch_label, debug_image, debug_label
        else:
            return batch_image, batch_label
    
    def on_epoch_end(self):
        self.current_epoch += 1
        if self.phase:
            self.dataset = shuffle(self.dataset)
