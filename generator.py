from typing import Dict, Callable, List

import datasets
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import additional_corruptions
import mnist_c

CORRUPTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Noise
    'shot_noise': lambda x: mnist_c.shot_noise(x),
    'impulse_noise': lambda x: mnist_c.impulse_noise(x),
    # Blur
    'glass_blur': lambda x: mnist_c.glass_blur(x),
    'motion_blur': lambda x: mnist_c.motion_blur(x),
    # Transformations
    'shear': lambda x: mnist_c.shear(x),
    'scale': lambda x: mnist_c.scale(x),
    'rotate': lambda x: mnist_c.rotate(x),
    'brightness': lambda x: mnist_c.brightness(x),
    'contrast': lambda x: mnist_c.contrast(x),
    'saturate': lambda x: mnist_c.saturate(x),
    'inverse': lambda x: mnist_c.inverse(x),
    # Turning and flipping
    'flip_sides': lambda x: additional_corruptions.flip_sides(x),
    'flip_up_down': lambda x: additional_corruptions.flip_up_down(x),
    'flip_left_right': lambda x: additional_corruptions.turn_right(x),
    'flip_top_bottom': lambda x: additional_corruptions.turn_left(x),
}

EXTENDED_CORRUPTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = \
    CORRUPTIONS.update(
        {
            'gaussian_noise': lambda x: mnist_c.gaussian_noise(x),
            'speckle_noise': lambda x: mnist_c.speckle_noise(x),
            'pessimal_noise': lambda x: mnist_c.pessimal_noise(x),
            'gaussian_blur': lambda x: mnist_c.gaussian_blur(x),
            'defocus_blur': lambda x: mnist_c.defocus_blur(x),
            'stripe': lambda x: mnist_c.stripe(x),
            'spatter': lambda x: mnist_c.spatter(x),
            'canny_edges': lambda x: mnist_c.canny_edges(x),
            'zoom_blur': lambda x: mnist_c.zoom_blur(x),
            'jpeg_compression': lambda x: mnist_c.jpeg_compression(x),
            'elastic_transform': lambda x: mnist_c.elastic_transform(x),
            'quantize': lambda x: mnist_c.quantize(x),
            'translate': lambda x: mnist_c.translate(x),
            'snow': lambda x: mnist_c.snow(x),
        }
    )


# REMOVED MNIIST_C CORRUPTIONS:
# 'fog': lambda x: mnist_c.fog(x),  # Removed as used GPL dependency
# 'frost': lambda x: mnist_c.frost(x),
# 'line': lambda x: mnist_c.line(x),
# 'pixelate': lambda x: mnist_c.pixelate(x),  # Aleady pixely enough
# 'identity': lambda x: mnist_c.identity(x),
# 'dotted_line': lambda x: mnist_c.dotted_line(x),
# 'zigzag': lambda x: mnist_c.zigzag(x),


def _random_corruption(rng: np.random.RandomState) -> str:
    return list(CORRUPTIONS.keys())[rng.randint(len(CORRUPTIONS))]


def generate_mix_dataset(imgs: List[List[List[int]]], split: str) -> np.ndarray:
    corrupted = []
    rng = np.random.RandomState()
    for img in tqdm(np.array(imgs)):
        img_as_array = np.array(img)
        corruption = CORRUPTIONS[_random_corruption(rng)]
        corrupted_img = corruption(img_as_array)
        corrupted.append(corrupted_img)
    imgs = np.array(corrupted).astype(np.uint8)
    np.save(f"generated/fmnist-c-{split}.npy", imgs)
    return imgs


def generate_for_corruption_type(seed: int, imgs: np.ndarray, corruption_type: str) -> np.ndarray:
    # TODO Not yet implemented
    pass


def generate_datasets(fmnist_split: str, seed_add=0):
    datasets.load_dataset_builder("fashion_mnist")
    dataset = load_dataset("fashion_mnist", split=fmnist_split)

    np.save(f"generated/fmnist-c-{fmnist_split}-labels.npy", np.array(dataset['label']))
    generate_mix_dataset(dataset['image'], fmnist_split)

    # TODO Generate for each corruption type



if __name__ == "__main__":
    generate_datasets(fmnist_split="train")
    generate_datasets(fmnist_split="test")
