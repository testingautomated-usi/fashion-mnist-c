import numpy as np
from matplotlib import pyplot as plt


def create_single_examples():
    # Load corrupted images
    imgs = np.load('./generated/npy/fmnist-c-test.npy')
    for i in range(8):
        inverted_img = 255 - imgs[i]
        plt.imsave(f"./generated/png-examples/single_{i}.png", inverted_img, cmap="gray")

    print("Created single images")

if __name__ == '__main__':
    create_single_examples()
