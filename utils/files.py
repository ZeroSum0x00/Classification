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


def verify_folder(folder):
    if folder[-1] != '/':
        folder += '/'
    return folder


def get_files(folder_path, extensions=['py', 'png', 'JPEG']):
    if isinstance(extensions, str):
        extensions = [extensions]
    else:
        extensions = [ex.lower() for ex in extensions]
    result = [x for x in os.listdir(folder_path) if x.split('.')[-1].lower() in extensions]
    return result

def valid_image(image_path):
    subprocess.run(f"mogrify {image_path}", shell=True)