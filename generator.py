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
    np.save(f"generated/npy/fmnist-c-{split}.npy", imgs)
    return imgs


def generate_datasets(fmnist_split: str):
    datasets.load_dataset_builder("fashion_mnist")
    dataset = load_dataset("fashion_mnist", split=fmnist_split)

    np.save(f"generated/npy/fmnist-c-{fmnist_split}-labels.npy", np.array(dataset['label']))
    generate_mix_dataset(dataset['image'], fmnist_split)


if __name__ == "__main__":
    generate_datasets(fmnist_split="train")
    generate_datasets(fmnist_split="test")
