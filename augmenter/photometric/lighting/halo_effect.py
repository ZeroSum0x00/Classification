import cv2
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.auxiliary_processing import is_numpy_image


def halo_effect(image, radius=0.5, alpha=0.8):

    def ring(x, y, h, w, max_dim, k_size_norm, ring_thickness_norm, radius):
        halo_ring = np.zeros((h, w), np.uint8)
        ring_thickness = int(max_dim * ring_thickness_norm)
        cv2.circle(halo_ring, (x, y), radius, 255, ring_thickness)

        k = int(max_dim * k_size_norm)
        k = k if k % 2 else k + 1
        return cv2.GaussianBlur(halo_ring.copy(), (k, k), 0)

    if not is_numpy_image(image):
        raise TypeError('img should be image. Got {}'.format(type(image)))
        
    h, w = image.shape[:2]
    avg_dim = (h + w) // 2
    radius = int(radius * avg_dim)

    x = random.randint(0, w - 1)
    y = random.randint(0, h - 1)

    halo_kernel = np.zeros((h, w), np.uint8)

    halo_kernel_radius = int(radius * random.uniform(.1, .35))
    cv2.circle(halo_kernel, (x, y), halo_kernel_radius, (255, 255, 255), -1)

    num_of_rays = 6
    b = halo_kernel_radius
    for _ in range(num_of_rays):
        offset_x = random.randint(b, int(2.5 * b))
        offset_y = random.randint(b, int(2.5 * b))

        offset_y = -offset_y if random.random() < .5 else offset_y
        offset_x = -offset_x if random.random() < .5 else offset_x

        cv2.line(halo_kernel, (x, y), (x + offset_x, y + offset_y), 255, 3)

    k1 = int(avg_dim * 0.1)
    halo_kernel = cv2.blur(halo_kernel, (k1, k1))
    halo_kernel = halo_kernel.astype(np.uint16)

    if random.random() < .5:
        halo_kernel += ring(x,
                            y,
                            h,
                            w,
                            avg_dim,
                            k_size_norm=random.uniform(.1, .25),
                            ring_thickness_norm=random.uniform(0.008, 0.015),
                            radius=radius)

    if random.random() < .5:
        halo_kernel += ring(x,
                            y,
                            h,
                            w,
                            avg_dim,
                            k_size_norm=random.uniform(.3, .5),
                            ring_thickness_norm=random.uniform(0.05, 0.2),
                            radius=radius)

    halo_kernel = np.clip(halo_kernel, 0, 255).astype(np.uint8)
    halo_kernel = cv2.cvtColor(halo_kernel, cv2.COLOR_GRAY2RGB)

    dst = cv2.addWeighted(image, 1.0, halo_kernel, alpha, 0.0)

    return cv2.resize(dst, (w, h), interpolation=cv2.INTER_CUBIC)


class HaloEffect(BaseTransform):
    def __init__(self, radius=0.5, alpha=0.8):
        self.radius = radius
        self.alpha  = alpha

    def image_transform(self, image):
        return halo_effect(image, self.radius, self.alpha)

class RandomHaloEffect(BaseRandomTransform):
    def __init__(self, radius=0.5, alpha=0.8, prob=0.5):
        self.radius = radius
        self.alpha  = alpha
        self.prob   = prob

    def image_transform(self, image):
        return halo_effect(image, self.radius, self.alpha)