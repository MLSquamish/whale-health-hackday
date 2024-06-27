from abc import abstractmethod, ABCMeta
from segment_anything import SamPredictor, sam_model_registry
import cv2

from whale_health.visualization_tools import get_mask_image
import numpy as np
from sklearn.cluster import KMeans


"""
Demo segment anything model:

pip install git+https://github.com/facebookresearch/segment-anything.git

Note: Not working now - segments in a super wierd format...
"""


class ISegmenter(metaclass=ABCMeta):

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """

        :param image: A BGR image as a numpy array (HxWx3)
        :return: A (n_channels, height, width) array of boolean segment masks
        """
        pass


class SegmentAnythingSegmenter(ISegmenter):

    def __init__(self, model_name: str = 'vit_h') -> None:
        self.model = sam_model_registry[model_name]()
        self.predictor = SamPredictor(self.model)

    def segment(self, image: np.ndarray) -> np.ndarray:
        self.predictor.set_image(image, image_format='BGR')
        masks, _, _ = self.predictor.predict()
        return masks


class KMColorSegmenter(ISegmenter):

    def __init__(self, n_colors: int = 3) -> None:
        self.n_colors = n_colors
        self.km = KMeans(n_clusters=n_colors)

    def segment(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        km_image = image.reshape(-1, 3)
        km_image = self.km.fit_predict(km_image).reshape(h, w)
        masks = np.zeros((self.n_colors, h, w), dtype=bool)
        for i in range(self.n_colors):
            masks[i] = km_image == i
        return masks





def demo_segment_anything(
        image_path = '/Users/peter/Downloads/Ha25/Ha25_5.png',
):
    image = cv2.imread(image_path)

    # Resize to 800x600
    image = cv2.resize(image, (800, 600))

    # sam = sam_model_registry["vit_h"]()
    # predictor = SamPredictor(sam)
    # predictor.set_image(image, image_format='BGR')
    # print("Segmenting Whale...")
    # masks, a, b = predictor.predict()

    masks = KMColorSegmenter(n_colors=2).segment(image)


    cv2.imshow('Raw', image)
    # display_masks = [(m*255).astype(np.uint8) for m in masks]
    #
    # for i, d in enumerate(display_masks):
    #
    #     merged = cv2.addWeighted(image.mean(axis=2).astype(np.uint8), 0.5, d, 0.5, 0)
    #     cv2.imshow(f'Segmented_{i}', merged)

    mask_diplay = get_mask_image(masks)

    merged = cv2.addWeighted(image.mean(axis=2, keepdims=True)[:, :, [0, 0, 0]].astype(np.uint8), 0.5, mask_diplay, 0.5, 0)
    cv2.imshow('Segmented', merged)

    # cv2.imshow('Merged', merged)
    cv2.waitKey(100000)


if __name__ == '__main__':
    demo_segment_anything()
