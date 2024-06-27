
import cv2
import numpy as np

def get_mask_image(masks):
    """
    Show image with segmentation overlay
    :param image: A BGR image as a numpy array
    :param masks: An (n_channels, height, width) array of boolean segment masks
    :return:
    """

    colors = np.array([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 0, 0],
        [255, 255, 255],
    ], dtype=np.uint8)

    image = np.sum(colors[:len(masks), None, None, :] * masks[:, :, :, None], axis=0) / np.sum(masks, axis=0)[:, :, None]
    return image.astype(np.uint8)


