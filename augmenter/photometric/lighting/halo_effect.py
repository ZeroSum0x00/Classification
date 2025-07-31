import cv2
import copy
import random
import numpy as np

from augmenter.base_transform import BaseTransform, BaseRandomTransform
from utils.augmenter_processing import extract_metadata, get_focus_image_from_metadata



def halo_effect(metadata, radius=0.5, alpha=0.8):

    def ring(x, y, h, w, max_dim, k_size_norm, ring_thickness_norm, radius):
        halo_ring = np.zeros((h, w), np.uint8)
        ring_thickness = int(max_dim * ring_thickness_norm)
        cv2.circle(halo_ring, (x, y), radius, 255, ring_thickness)

        k = int(max_dim * k_size_norm)
        k = k if k % 2 else k + 1
        return cv2.GaussianBlur(halo_ring.copy(), (k, k), 0)

    if isinstance(metadata, dict):
        metadata_check = True
        clone_data = copy.deepcopy(metadata)
        algorithm, image_data, auxi_image_data, masks_data, bbox_data, landmark_data = extract_metadata(clone_data)
        image = image_data.get("value")
    elif isinstance(metadata, np.ndarray):
        image = copy.deepcopy(metadata)
        metadata_check = False
    else:
        raise ValueError("Input must be either a dictionary (metadata) or a NumPy array (image).")
        
    if image is None:
        return metadata

    height, width = image.shape[:2]
    avg_dim = (height + width) // 2
    radius = int(radius * avg_dim)

    pos_x = random.randint(0, width - 1)
    pos_y = random.randint(0, height - 1)

    halo_kernel = np.zeros((height, width), np.uint8)

    halo_kernel_radius = int(radius * random.uniform(.1, .35))
    cv2.circle(halo_kernel, (pos_x, pos_y), halo_kernel_radius, (255, 255, 255), -1)

    num_of_rays = 6
    b = halo_kernel_radius
    for _ in range(num_of_rays):
        offset_x = random.randint(b, int(2.5 * b))
        offset_y = random.randint(b, int(2.5 * b))

        offset_y = -offset_y if random.random() < .5 else offset_y
        offset_x = -offset_x if random.random() < .5 else offset_x

        cv2.line(halo_kernel, (pos_x, pos_y), (pos_x + offset_x, pos_y + offset_y), 255, 3)

    k1 = int(avg_dim * 0.1)
    halo_kernel = cv2.blur(halo_kernel, (k1, k1))
    halo_kernel = halo_kernel.astype(np.uint16)

    if random.random() < .5:
        halo_kernel += ring(
            x=pos_x,
            y=pos_y,
            h=height,
            w=width,
            max_dim=avg_dim,
            k_size_norm=random.uniform(.1, .25),
            ring_thickness_norm=random.uniform(0.008, 0.015),
            radius=radius,
        )

    if random.random() < .5:
        halo_kernel += ring(
            x=pos_x,
            y=pos_y,
            h=height,
            w=width,
            max_dim=avg_dim,
            k_size_norm=random.uniform(.3, .5),
            ring_thickness_norm=random.uniform(0.05, 0.2),
            radius=radius,
        )

    halo_kernel = np.clip(halo_kernel, 0, 255).astype(np.uint8)
    halo_kernel_rgb = cv2.cvtColor(halo_kernel, cv2.COLOR_GRAY2RGB)
    image = cv2.addWeighted(image, 1.0, halo_kernel_rgb, alpha, 0.0)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    circle_mask = np.zeros((height, width), dtype=np.uint8)    
    circle_mask = cv2.addWeighted(circle_mask, 1.0, halo_kernel, alpha, 0.0)

    if metadata_check:
        clone_data["image"] = image
        
        if landmark_data.get("value") is not None:
            landmarks = copy.deepcopy(landmark_data.get("value"))
            clip_out_range = landmark_data.get("clip_out_range", False)
        
            if landmarks.ndim == 2 and landmarks.shape[1] == landmark.get("num_points", 3)*3:
                landmarks = landmarks.reshape(-1, m, 3)
            elif landmarks.ndim != 3 or landmarks.shape[2] != 3:
                raise ValueError(f"Landmark shape invalid: {landmarks.shape}")
            
            for i, points in enumerate(landmarks):
                for j, point in enumerate(points):
                    x, y, vis = point
                    
                    if x < 0 or x >= width:
                        continue
    
                    if y < 0 or y >= height:
                        continue
    
                    if vis == 0:
                        continue
    
                    if circle_mask[y, x] > 150:
                        landmarks[i, j, -1] = 0
                        
            clone_data["landmark_point"]["value"] = landmarks

        return clone_data
    else:
        return image


class HaloEffect(BaseTransform):
    def __init__(self, radius=0.5, alpha=0.8):
        self.radius = radius
        self.alpha = alpha

    def image_transform(self, metadata):
        return halo_effect(metadata, self.radius, self.alpha)


class RandomHaloEffect(BaseRandomTransform):
    def __init__(self, radius=0.5, alpha=0.8, prob=0.5):
        self.radius = radius
        self.alpha = alpha
        self.prob = prob

    def image_transform(self, metadata):
        return halo_effect(metadata, self.radius, self.alpha)
    