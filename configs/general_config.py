from augmenter import *

INPUT_SHAPE = (224, 224, 3)

DATA_PATH = "/home/vbpo-101386/Desktop/TuNIT/Datasets/Classification/PetImages"

DATA_DESTINATION_PATH = None

DATA_ANNOTATION_PATH = "./configs/classes.names"

DATA_COLOR_SPACE = 'rgb'

DATA_NORMALIZER = 'sub_divide'

DATA_MEAN_NORMALIZATION = None

DATA_STD_NORMALIZATION = None

DATA_AUGMENTATION = {
            "train": [
                        ResizePadded(INPUT_SHAPE, flexible=True, padding_color=128), 
                        RandomFlip(mode='horizontal'), 
                        RandomRotate(angle_range=20, prob=0.5, padding_color=128),
                        LightIntensityChange(),
            ],
            "valid": [ResizePadded(INPUT_SHAPE, flexible=False, padding_color=128)],
            "test": None
}

DATA_TYPE = None

CHECK_DATA = False

DATA_LOAD_MEMORY = False

TRAIN_BATCH_SIZE = 16

TRAIN_EPOCH_INIT = 0

TRAIN_EPOCH_END = 200

TRAIN_LR_INIT = 1e-2

TRAIN_LR_END = 1e-4

TRAIN_WARMUP_EPOCH_RATIO        = 0.05

TRAIN_WARMUP_LR_RATIO           = 0.1

WITHOUT_AUG_EPOCH_RATIO         = 0.05

TRAIN_WEIGHT_TYPE = None

TRAIN_WEIGHT_OBJECTS = [        
                                {
                                  'path': './saved_weights/20220926-100327/best_weights_mAP',
                                  'stage': 'full',
                                  'custom_objects': None
                                }
                              ]

TRAIN_RESULT_SHOW_FREQUENCY = 10

TRAIN_SAVE_WEIGHT_FREQUENCY = 100

TRAIN_SAVED_PATH = './saved_weights/'

TRAIN_MODE = 'graph'
