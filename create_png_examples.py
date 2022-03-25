import numpy as np
from matplotlib import pyplot as plt


def create_single_examples():
    # Load corrupted images
    imgs = np.load('./generated/npy/fmnist-c-test.npy')
    for i in range(8):
        inverted_img = 255 - imgs[i]
        plt.imsave(f"./generated/png-examples/single_{i}.png", inverted_img, cmap="gray")

    print("Created single images")


ROWS_PER_CLASS = 3
EXAMPLES_PER_ROW = 30


def create_multi_example():
    imgs = np.load('./generated/npy/fmnist-c-test.npy')
    labels = np.load('./generated/npy/fmnist-c-test-labels.npy')

    chosen_imgs = []
    for c in range(10):
        # Create a list of images with the same label
        c_imgs = imgs[labels == c]
        for i in range(ROWS_PER_CLASS):
            chosen_imgs.append(c_imgs[i * EXAMPLES_PER_ROW: (i + 1) * EXAMPLES_PER_ROW])

    # Create a grid of images
    grid = np.concatenate(chosen_imgs, axis=1)
    # TODO flatten first axis to become single image
    # TODO plot image


if __name__ == '__main__':
    create_single_examples()
