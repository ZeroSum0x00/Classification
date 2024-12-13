import os
import cv2
import imagesize
from tqdm import tqdm
from utils.files import valid_image


class ParseDirName:
    def __init__(self, 
                 data_dir,
                 classes,
                 load_memory,
                 check_data):
        self.data_dir    = data_dir
        self.classes     = classes
        self.load_memory = load_memory
        self.check_data  = check_data

    def __call__(self, image_files):
        data_extraction = []

        for filename in tqdm(image_files):
            info_dict = {}
            info_dict['filename'] = os.path.basename(filename)
            info_dict['image'] = None
            label = os.path.dirname(filename)
            info_dict['label'] = self.classes.index(label)
            info_dict['path'] = os.path.join(self.data_dir, label)
            image_path = os.path.join(self.data_dir, filename)
            width, height = imagesize.get(image_path)
            info_dict['image_size'] = (height, width)

            if self.check_data:
                try:
                    valid_image(image_path)
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    shape = img.shape
                except Exception as e:
                    print(f"Error: File {filename} is can't loaded: {e}")
                    continue
                
            if self.load_memory:
                img = cv2.imread(image_path)
                info_dict['image'] = img
                
            if info_dict:
                data_extraction.append(info_dict)
        return data_extraction