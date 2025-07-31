import numpy as np



def get_focus_image_from_metadata(metadata):
    focus_image = None

    if isinstance(metadata, dict):
        image = metadata.get("image")
        auxi_image = metadata.get("auxiliary_images", {})
        masks = metadata.get("masks", {})

        if image is not None:
            focus_image = image
        elif any(array is not None for array in auxi_image.values()):
            for array in auxi_image.values():
                if array is not None and isinstance(array, np.ndarray):
                    focus_image = array                    
                    break
        elif any(array is not None for array in masks.values()):
            for array in masks.values():
                if array is not None and isinstance(array, np.ndarray):                    
                    focus_image = array                    
                    break
                    
    elif isinstance(metadata, np.ndarray):
        focus_image = metadata
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")

    if focus_image is None:
        raise TypeError("No valid image or mask found for determining shape.")

    return focus_image


def extract_metadata(metadata):
    algorithm = metadata.get("algorithm")
    image = metadata.get("image")
    auxiliary_images = metadata.get("auxiliary_images", {})
    masks = metadata.get("masks", {})
    bounding_box = metadata.get("bounding_box", {})
    landmark_point = metadata.get("landmark_point", {})
    return algorithm, image, auxiliary_images, masks, bounding_box, landmark_point
