from abc import abstractmethod, ABCMeta
from segment_anything import SamPredictor, sam_model_registry
import cv2

from whale_health.visualization_tools import get_mask_image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


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


class HandCodedColorSegmenter(ISegmenter):
    """
    We are looking for greyish whale colour, without the blue of the water or the white of the reflection.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        masks = np.zeros((2, h, w), dtype=bool)
        masks[1] = np.abs(image[:, :, 0] - image[:, :, 1]) < 10
        masks[1][image.mean(axis=2) > 175] = False
        # masks[1] = np.abs(image[:, :, 0] - image[:, :, 2]) < 10
        # masks[2] = np.abs(image[:, :, 1] - image[:, :, 2]) < 10
        masks[0] = np.logical_not(np.any(masks[1:], axis=0))
        return masks
#
#
# def locally_normalize_image(image: np.ndarray, box_filter_relative_width: float = 0.2) -> np.ndarray:
#     """
#     Locally normalize an image by subtracting the local mean and dividing by the local standard deviation.
#     :param image: A BGR image as a numpy array
#     :param box_filter_relative_width: The relative width of the box filter used to compute the local mean and std
#     :return: The locally normalized image
#     """
#     h, w, _ = image.shape
#     box_filter_width = int(box_filter_relative_width * min(h, w))
#     image = image.astype(np.float32)
#     mean = cv2.boxFilter(image, -1, (box_filter_width, box_filter_width), normalize=False)





def demo_segment_anything(
        image_path = '/Users/peter/Downloads/Ha25/Ha25_5.png',
):
    image = cv2.imread(image_path)

    # Resize to 800x600
    # image = cv2.resize(image, (640, 480))
    image = cv2.resize(image, (1920, 1080))

    # image = locally_normalize_image(image)


    # sam = sam_model_registry["vit_h"]()
    # predictor = SamPredictor(sam)
    # predictor.set_image(image, image_format='BGR')
    # print("Segmenting Whale...")
    # masks, a, b = predictor.predict()

    # masks = KMColorSegmenter(n_colors=2).segment(image)
    masks = HandCodedColorSegmenter().segment(image)
    # masks = DBScanColorSegmenter(eps=0.5, min_samples=5).segment(image)


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
