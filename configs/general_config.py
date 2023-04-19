# import config parameters
# from augmenter import *
# from utils.post_processing import get_labels


CLASSES_FILE                    = './configs/classes.names'

# YOLO_CLASSES, NUM_CLASSES       = get_labels(CLASSES_FILE)

# YOLO hyper-parameters
YOLO_ARCHITECTURE               = 'yolov3'

YOLO_ACTIVATION                 = 'leaky'

YOLO_NORMALIZATION              = 'batchnorm'

YOLO_BACKBONE_ACTIVATION        = 'leaky'

YOLO_BACKBONE_NORMALIZATION     = 'batchnorm'

YOLO_BACKBONE_WEIGHTS           = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/saved_weights/yolov3.weights"

YOLO_TARGET_SIZE                = [416, 416, 3]

YOLO_ANCHORS                    = [[ 10,  13],
                                   [ 16,  30],
                                   [ 33,  23],
                                   [ 30,  61],
                                   [ 62,  45],
                                   [ 59, 119],
                                   [116,  90],
                                   [156, 198],
                                   [373, 326]]

YOLO_ANCHORS_MASK               = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

YOLO_STRIDES                    = [8, 16, 32]

YOLO_MAX_BBOXES                 = 100

YOLO_IGNORE_THRESHOLD           = 0.5

YOLO_BALANCE_LOSS               = [0.4, 1.0, 4]

YOLO_BOX_RATIO_LOSS             = 0.05

# YOLO_OBJ_RATIO_LOSS       /      = 5 * (YOLO_TARGET_SIZE[0] * YOLO_TARGET_SIZE[1]) / (416 ** 2)

# YOLO_CLS_RATIO_LOSS             = 1 * (NUM_CLASSES / 80)

YOLO_LABEL_SMOOTHING            = 0.1

YOLO_IOU_METHOD                 = 'CIOU'

YOLO_FOCAL_LOSS                 = True

YOLO_FOCAL_LOSS_RATIO           = 10

YOLO_FOCAL_ALPHA_RATIO          = 0.25

YOLO_FOCAL_GAMMA_RATIO          = 2


# Training hyper-parameters
DATA_PATH                       = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/datasets/VOC2017"

DATA_ANNOTATION_PATH            = None

DATA_DESTINATION_PATH           = None

DATA_COLOR_SPACE                = 'RGB'

DATA_COORDINATE                 = "corners"

# DATA_AUGMENTATION               = {
#                                       'train': {
#                                           'main': [
#                                               ResizePadded(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES, jitter=.3, flexible=True),
#                                               RandomFlip(coords=DATA_COORDINATE, mode='horizontal'),
#                                               LightIntensityChange(hue=.1, sat=0.7, val=0.4, color_space=DATA_COLOR_SPACE),
#                                           ],
#                                           'auxiliary': None,
#                                           'merge': None
#                                       },
#                                       'valid': {
#                                           'main': [ResizePadded(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES, jitter=.3, flexible=False)],
#                                           'auxiliary': None,
#                                           'merge': None
#                                       },
#                                       'test': {
#                                           'main': [ResizePadded(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES, jitter=.3, flexible=False)],
#                                           'auxiliary': None,
#                                           'merge': None
#                                       }
# }

# DATA_ENDEMIC_AUGMENTATION       = {
#                                       'train': {
#                                           'main': [
#                                               Mosaic(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES),
#                                               RandomFlip(coords=DATA_COORDINATE, mode='horizontal'),
#                                               LightIntensityChange(hue=.1, sat=0.7, val=0.4, color_space=DATA_COLOR_SPACE),
#                                           ],
#                                           'auxiliary': [ResizePadded(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES, jitter=.3, flexible=True)],
#                                           'merge': [Mixup(target_size=YOLO_TARGET_SIZE, coords=DATA_COORDINATE, max_bboxes=YOLO_MAX_BBOXES)]
#                                       },
#                                       'valid': None,
#                                       'test':  None
# }

DATA_ENDEMIC_AUGMENTATION_PROBA = 0.5

DATA_ENDEMIC_AUGMENTATION_RATIO = 0.7

DATA_NORMALIZER                 = 'divide'

DATA_TYPE                       = 'voc'

CHECK_DATA                      = False

DATA_LOAD_MEMORY                = False

DATA_EXCLUDE_CROWD              = True

DATA_EXCLUDE_DIFFICULT          = True

DATA_EXCLUDE_TRUNCATED          = False

TRAIN_BATCH_SIZE                = 8

TRAIN_EPOCH_INIT                = 0

TRAIN_EPOCH_END                 = 300

TRAIN_OPTIMIZER                 = 'sgd'

TRAIN_MOMENTUM                  = 0.937

TRAIN_NESTEROV                  = True

TRAIN_LR_INIT                   = 1e-2

TRAIN_LR_END                    = 1e-4

TRAIN_WARMUP_EPOCH_RATIO        = 0.05

TRAIN_WARMUP_LR_RATIO           = 0.1

WITHOUT_AUG_EPOCH_RATIO         = 0.05

TRAIN_WEIGHT_TYPE               = None

TRAIN_WEIGHT_OBJECTS            = [        
                                    {
                                      'path': './saved_weights/20220926-100327/best_weights_mAP',
                                      'stage': 'full',
                                      'custom_objects': None
                                    }
                                  ]

TRAIN_RESULT_SHOW_FREQUENCY     = 10

TRAIN_SAVE_WEIGHT_FREQUENCY     = 50

TRAIN_SAVED_PATH                = './saved_weights/'

TRAIN_MODE                      = 'graph'


# Inference (validation-testing-predict) hyper-parameters
TEST_CONFIDENCE_THRESHOLD       = 0.05

TEST_IOU_THRESHOLD              = 0.5

TEST_MIN_OVERLAP                = 0.5