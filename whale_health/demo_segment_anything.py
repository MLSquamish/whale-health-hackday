
from segment_anything import SamPredictor, sam_model_registry
import cv2

from whale_health.visualization_tools import get_mask_image
import numpy as np


"""
Demo segment anything model:

pip install git+https://github.com/facebookresearch/segment-anything.git

Note: Not working now - segments in a super wierd format...
"""


def demo_segment_anything(
        image_path = '/Users/peter/Downloads/Ha25/Ha25_5.png',
):
    image = cv2.imread(image_path)

    # Resize to 800x600
    image = cv2.resize(image, (800, 600))

    sam = sam_model_registry["vit_h"]()
    predictor = SamPredictor(sam)
    predictor.set_image(image, image_format='BGR')
    print("Segmenting Whale...")
    masks, a, b = predictor.predict()

    cv2.imshow('Raw', image)
    display_masks = [(m*255).astype(np.uint8) for m in masks]

    for i, d in enumerate(display_masks):
        merged = cv2.addWeighted(image.mean(axis=2).astype(np.uint8), 0.5, d, 0.5, 0)
        cv2.imshow(f'Segmented_{i}', merged)

    # cv2.imshow('Merged', merged)
    cv2.waitKey(100000)


if __name__ == '__main__':
    demo_segment_anything()
