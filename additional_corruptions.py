"""The corruptions which are not part of mnist-c."""

import numpy as np


def flip_sides(image):
    """Flip the image left to right."""
    return np.fliplr(image)


def flip_up_down(image):
    """Flip the image upside down."""
    return np.flipud(image)


def turn_left(image):
    """Turn the image 90 degrees to the left."""
    return np.rot90(image)


def turn_right(image):
    """Turn the image 90 degrees to the right."""
    return np.rot90(image, 3)
