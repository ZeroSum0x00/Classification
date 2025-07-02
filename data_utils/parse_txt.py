import os
import cv2
from tqdm import tqdm



class ParseTXT:
    def __init__(self, 
        data_dir,
        annotation_file,
        classes,
        load_memory,
        check_data,
        *args, **kwargs
    ):
        self.data_dir = data_dir
        self.annotation_path = annotation_file
        txt_file = open(annotation_file, "r")
        self.raw_data = txt_file.readlines()
        txt_file.close()

        self.classes = classes
        self.load_memory = load_memory
        self.check_data = check_data

    def __call__(self):
        data_extraction = []
        for line in tqdm(self.raw_data, desc="Load dataset"):
            info_dict = {}
            image_path, label = line.strip().split('\t')

            folder, filename = os.path.split(image_path)
            info_dict['filename'] = filename
            info_dict['image'] = None
            info_dict['label'] = self.classes.index(label)
            info_dict['path'] = folder

            if self.check_data:
                try:
                    valid_image(image_path)
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    shape = img.shape
                except Exception as e:
                    logger.error(f"Error: File {filename} is can't loaded: {e}")
                    continue
                    
            if self.load_memory:
                img = cv2.imread(image_path)
                info_dict["image"] = img
                
            if info_dict:
                data_extraction.append(info_dict)
        return data_extraction
