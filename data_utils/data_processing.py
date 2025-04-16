import os
from utils.files import  get_files
from utils.post_processing import get_labels
from utils.constants import ALLOW_IMAGE_EXTENSIONS
from utils.logger import logger
from data_utils import ParseDirName, DataSequencePipeline, TFDataPipeline


def get_data(
    data_dirs,
    classes,
    data_type=None,
    phase='train',
    check_data=False,
    load_memory=False,
    *args, **kwargs
):

    def load_data(data_dir):
        if data_type.lower() == "dirname":
            image_file_list = [sorted(get_files(os.path.join(data_dir, cls), ALLOW_IMAGE_EXTENSIONS, cls)) for cls in classes]
            image_files = [item for sublist in image_file_list for item in sublist]
            parser = ParseDirName(data_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
            return parser(image_files)

    assert data_type.lower() in ('dirname')
    data_extraction = []

    if isinstance(data_dirs, (list, tuple)):
        for data_dir in data_dirs:
            data_dir = os.path.join(data_dir, phase)
            parser = load_data(data_dir)
            data_extraction.extend(parser)
    else:
        data_dir = os.path.join(data_dirs, phase)
        parser = load_data(data_dir)
        data_extraction.extend(parser)
        
    return data_extraction


def get_train_test_data(
    data_dirs, 
    classes=None, 
    target_size=[224, 224, 3], 
    batch_size=16, 
    color_space='RGB',
    augmentor=None,
    normalizer=None,
    mean_norm=None,
    std_norm=None,
    interpolation="BILINEAR",
    data_type='dirname',
    check_data=False, 
    load_memory=False,
    dataloader_mode='tf',
    get_data_mode=0,
    num_workers=1,
    *args, **kwargs
):
    """
        get_data_mode = 0:   train - validation - test
        get_data_mode = 1:   train - validation
        get_data_mode = 2:   train
    """
    data_args = {
        "target_size": target_size,
         "batch_size": batch_size,
         "color_space": color_space,
         "augmentor": augmentor,
         "normalizer": normalizer,
         "mean_norm": mean_norm,
         "std_norm": std_norm,
         "interpolation": interpolation,
         "num_workers": num_workers,
        **kwargs
    }
    
    if classes:
        if isinstance(classes, str):
            classes, _ = get_labels(classes)
    else:
        classes, _ = get_labels(data_dirs)

    data_train = get_data(
        data_dirs,
        classes=classes,
        data_type=data_type,
        phase='train', 
        check_data=check_data,
        load_memory=load_memory,
    )
    
    train_args = {"dataset": data_train, "phase": "train", **data_args}
    train_generator = TFDataPipeline(**train_args) if dataloader_mode.lower() == 'tf' else DataSequencePipeline(**train_args)

    if get_data_mode != 2:
        data_valid = get_data(
            data_dirs,
            classes=classes,
            data_type=data_type,
            phase='validation', 
            check_data=check_data,
            load_memory=load_memory,
        )
        
        valid_args = {"dataset": data_valid, "phase": "valid", **data_args}
        valid_generator = TFDataPipeline(**valid_args) if dataloader_mode.lower() == 'tf' else DataSequencePipeline(**valid_args)
    else:
        valid_generator = None
        
    if get_data_mode == 0:
        data_test = get_data(
            data_dirs,
            classes=classes,
            data_type=data_type,
            phase='test', 
            check_data=check_data,
            load_memory=load_memory,
        )
        
        test_args = {"dataset": data_test, "phase": "test", **data_args}
        test_generator = TFDataPipeline(**test_args) if dataloader_mode.lower() == 'tf' else DataSequencePipeline(**test_args)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator