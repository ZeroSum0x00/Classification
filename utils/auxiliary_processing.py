import io
import cv2
import importlib
import numpy as np



def dynamic_import(module_name, global_vars=None):
    components = module_name.split(".")
    target = components[-1]
    
    if global_vars is None:
        global_vars = globals()

    if target in global_vars:
        return global_vars[target]
    
    if len(components) > 1:
        module = importlib.import_module(".".join(components[:-1]))
    else:
        module = importlib.import_module(__name__)
        
    try:
        return getattr(module, target)
    except AttributeError:
        raise ImportError(f"'{target}' does not exist in module '{'.'.join(components[:-1])}'")


def random_to_negative(number):
    if np.random.choice([0, 1]):
        return number
    else:
        return -number


def random_range(a=0, b=1):
    return np.random.rand() * (b - a) + a


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

    
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
    
    
def change_color_space(image, current_space="BGR", to_space="BGR"):
    if not ((current_space.lower() in {"bgr", "rgb", "hsv", "gray"}) 
            and (to_space.lower() in {"bgr", "rgb", "hsv", "gray"})):
        raise NotImplementedError
    if current_space.lower() != "gray" or (image.shape[-1] != 1 and len(image.shape) > 2):
        if current_space.lower() == "bgr" and to_space.lower() == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif current_space.lower() == "rgb" and to_space.lower() == "bgr":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif current_space.lower() == "bgr" and to_space.lower() == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif current_space.lower() == "hsv" and to_space.lower() == "bgr":
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif current_space.lower() == "rgb" and to_space.lower() == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif current_space.lower() == "hsv" and to_space.lower() == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif current_space.lower() == "bgr" and to_space.lower() == "gray":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif current_space.lower() == "rgb" and to_space.lower() == "gray":
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif current_space.lower() == "hsv" and to_space.lower() == "gray":
            return cv2.cvtColor(image, cv2.COLOR_HSV2GRAY)
    return image


def fig_to_cv2_image(fig):
    """
    Chuyển matplotlib Figure sang ảnh OpenCV.
    
    Args:
        fig (matplotlib.figure.Figure): Figure cần chuyển.
        
    Returns:
        image_cv2 (np.ndarray): Ảnh BGR.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    image_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image_cv2 = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    buf.close()
    return image_cv2