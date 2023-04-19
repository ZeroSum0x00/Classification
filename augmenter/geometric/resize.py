import cv2
import random
import numpy as np
from utils.auxiliary_processing import random_range


class ResizePadded:
    def __init__(self, target_size=(224, 224, 3), jitter=.3, flexible=False, padding_color=None):
        self.target_size = target_size
        self.jitter      = jitter
        self.flexible    = flexible
        self.padding_color = padding_color

    def __call__(self, image):
        h, w, _    = image.shape
        ih, iw, _  = self.target_size
        fill_color  = self.padding_color if self.padding_color else [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        
        if not self.flexible:
            scale = min(iw/w, ih/h)
            nw, nh  = int(scale * w), int(scale * h)
            dw, dh = (iw - nw) // 2, (ih - nh) // 2
            image_resized = cv2.resize(image, (nw, nh))
            image_paded = np.full(shape=[ih, iw, 3], fill_value=fill_color, dtype=image.dtype)
            image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
            return image_paded

        new_ar = w / h * random_range(1 - self.jitter, 1 + self.jitter) / random_range(1 - self.jitter, 1 + self.jitter)
        scale = random_range(0.75, 1.5)

        if new_ar < 1:
            nh = int(scale * ih)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * iw)
            nh = int(nw / new_ar)
          
        dw = int(random_range(0, iw - nw))
        dh = int(random_range(0, ih - nh))

        image_resized = cv2.resize(image, (nw, nh))

        height = max(ih, nh + abs(dh))
        width = max(iw, nw + abs(dw))
        image_paded = np.full(shape=[height, width, 3], fill_value=fill_color, dtype=image.dtype)
        if dw < 0 and dh >= 0:
            image_paded[dh:nh+dh, 0:nw, :] = image_resized
            if width == iw:
                image_paded = image_paded[:ih, :iw]
            else:
                image_paded = image_paded[:ih, abs(dw):abs(dw)+iw]
        elif dh < 0 and dw >= 0:
            image_paded[0:nh, dw:dw+nw, :] = image_resized
            if height == ih:
                image_paded = image_paded[:ih, :iw]
            else:
                image_paded = image_paded[abs(dh):abs(dh)+ih, :iw]
        elif dh < 0 and dw < 0:
            image_paded[0:nh, 0:nw, :] = image_resized
            if width == iw or height == ih:
                image_paded = image_paded[:ih, :iw]
            else:
                image_paded = image_paded[abs(dh):abs(dh)+ih, abs(dw):abs(dw)+iw]
        else:
            image_paded[dh:nh+dh, dw:dw+nw, :] = image_resized
            image_paded = image_paded[:ih, :iw]

        hpd, wpd, _ = image_paded.shape
        if hpd < ih or wpd <iw:
            image_temp = np.full(shape=[ih, iw, 3], fill_value=128.0)
            image_temp[:hpd, :wpd] = image_paded
            image_paded = image_temp

        image = image_paded
        image_data      = np.array(image, np.uint8)
        return image_data
