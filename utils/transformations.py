"""
Module for defining transformations for image processing.
"""
import random
from torchvision.transforms import functional as F

# pylint: disable=too-few-public-methods
class Compose:
    """
    Composes several transformations together.

    Parameters:
    - transforms (list): List of transformations to compose.

    Returns:
    - tuple: Tuple containing the transformed image and target.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """
    Randomly flips the input image horizontally with a given probability.

    Parameters:
    - prob (float): Probability of horizontal flip.

    Returns:
    - tuple: Tuple containing the transformed image and target.
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor:
    """
    Converts the input image to a PyTorch tensor.

    Returns:
    - tuple: Tuple containing the transformed image and target.
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
