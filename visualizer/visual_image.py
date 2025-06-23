import cv2
import numpy as np
from matplotlib import pyplot as plt



def visual_image(imgs, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    figure = plt.figure(figsize=size)
    show_imgs = [imgs] if not isinstance(imgs, list) else imgs

    if titles is not None:
        show_titles = [titles] if not isinstance(titles, list) else titles
    else:
        if not isinstance(imgs, list):
            show_titles = ["show screen"]
        else:
            show_titles = [idx for idx in range(len(imgs))]

    for index, (img, title) in enumerate(zip(show_imgs, show_titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if mode and mode.lower() == "bgr2rgb":
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)


def draw_boxes_on_image(image, boxes):
    for box in boxes:
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=(255, 0, 0),
            thickness=2,
        )
    image = np.clip(image, 0, 1)
    return image


def visual_image_with_bboxes(images, bboxes=None, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    images_copied = np.copy(images)
    bboxes_copied = np.copy(bboxes)
    figure = plt.figure(figsize=size)

    for index, (img, bbox, title) in enumerate(zip(images_copied, bboxes_copied, titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if bbox is not None:
            img = draw_boxes_on_image(img, bbox)

        if mode and mode.lower() == "bgr2rgb":
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)

    plt.show()
