"""A simple module used extract some sample images as PNG files.

The module does not actually create any new corrupted images,
it just loads them from the .npy files and formats them as PNG."""

import numpy as np
from matplotlib import pyplot as plt


def create_single_examples():
    """Stores 8 images as png, each as a single (i.e., individual) image."""
    # Load corrupted images
    imgs = np.load("./generated/npy/fmnist-c-test.npy")
    for i in range(8):
        inverted_img = 255 - imgs[i]
        plt.imsave(
            f"./generated/png-examples/single_{i}.png", inverted_img, cmap="gray"
        )

    print("Created single images")


if __name__ == "__main__":
    create_single_examples()
