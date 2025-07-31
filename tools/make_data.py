import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natsort import natsorted
from utils.files import get_files
from utils.constants import ALLOW_IMAGE_EXTENSIONS



if __name__ == "__main__":
    datasets_path = "/mnt/data_disk/Datasets/Classification/full_animals"
    output_path = "./datasets"

    data_folders = [name for name in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, name))]

    for folder in data_folders:
        result_file = open(os.path.join(output_path, f"{folder}.txt"), 'w')
        subpath = os.path.join(datasets_path, folder)

        if not os.path.isdir(subpath):
            continue

        image_path = natsorted(os.listdir(subpath))

        for path in image_path:
            filename_list = get_files(os.path.join(subpath, path), ALLOW_IMAGE_EXTENSIONS)

            for filename in filename_list:
                result_file.write(f"{os.path.join(subpath, path, filename)}\t{path}\n")

        result_file.close()
