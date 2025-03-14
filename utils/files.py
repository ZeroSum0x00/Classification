import os
import subprocess
from utils.logger import logger


def extract_zip(archive, destination):
    import patoolib
    try:
        if not os.path.exists(destination):
            os.mkdir(destination)
        logger.info("Extract files to {}".format(destination))
        patoolib.extract_archive(archive, outdir=destination, verbosity=-1)
        logger.info("Extract Done!")
    except BaseException as e:
        logger.error("Error: ", e)


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = os.path.join(data_destination, folder_name) 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def verify_folder(folder):
    if folder[-1] != '/':
        folder += '/'
    return folder


def get_files(folder_path, extensions=['py', 'png', 'JPEG'], prefix=''):
    if isinstance(extensions, str):
        extensions = [extensions]
    else:
        extensions = [ex.lower() for ex in extensions]
    
    result = []
    
    if os.path.isdir(folder_path):
        result = [os.path.join(prefix, x) for x in os.listdir(folder_path) if x.split('.')[-1].lower() in extensions]
        
    return result


def valid_image(image_path):
    subprocess.run(f"mogrify {image_path}", shell=True)