"""Corrupted Fashion-Mnist Data Set.

This module contains the huggingface dataset adaptation of
the Corrupted Fashion-Mnist Data Set.
Find the full code at `https://github.com/testingautomated-usi/fashion-mnist-c`."""
import struct

import datasets
import numpy as np
from datasets.tasks import ImageClassification

_CITATION = """\
@inproceedings{Weiss2022SimpleTechniques,
  title={Simple Techniques Work Surprisingly Well for Neural Network Test Prioritization and Active Learning},
  author={Weiss, Michael and Tonella, Paolo},
  booktitle={Proceedings of the 31th ACM SIGSOFT International Symposium on Software Testing and Analysis},
  year={2022}
}
"""

_DESCRIPTION = """\
Fashion-MNIST is dataset of fashion images, indended as a drop-in replacement for the MNIST dataset.
This dataset (Fashion-Mnist-Corrupted) provides out-of-distribution data for the Fashion-Mnist
dataset. Fashion-Mnist-Corrupted is based on a similar project for MNIST, called MNIST-C, by Mu et. al.
"""

CONFIG = datasets.BuilderConfig(
    name="fashion_mnist_corrupted",
    version=datasets.Version("1.0.0"),
    description=_DESCRIPTION,
)

_HOMEPAGE = "https://github.com/testingautomated-usi/fashion-mnist-c"
_LICENSE = "https://github.com/testingautomated-usi/fashion-mnist-c/blob/main/LICENSE"

if CONFIG.version == datasets.Version("1.0.0"):
    tag = "v1.0.0"
else:
    raise ValueError("Unsupported version.")

_URL = (
    f"https://github.com/testingautomated-usi/fashion-mnist-c/blob/{tag}/generated/npy/"
)

_URLS = {
    "train_images": "fmnist-c-train.npy",
    "train_labels": "fmnist-c-train-labels.npy",
    "test_images": "fmnist-c-test.npy",
    "test_labels": "fmnist-c-test-labels.npy",
}

_NAMES = [
    "T - shirt / top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class FashionMnistCorrupted(datasets.GeneratorBasedBuilder):
    """FashionMNIST-Corrupted Data Set"""

    BUILDER_CONFIGS = [CONFIG]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[
                ImageClassification(image_column="image", label_column="label")
            ],
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            key: _URL + fname + "?raw=true" for key, fname in _URLS.items()
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": [
                        downloaded_files["train_images"],
                        downloaded_files["train_labels"],
                    ],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": [
                        downloaded_files["test_images"],
                        downloaded_files["test_labels"],
                    ],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw form."""
        # Images
        images = np.load(filepath[0])
        labels = np.load(filepath[1])

        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of images {images.shape[0]} and labels {labels.shape[0]} do not match."
            )

        for idx in range(images.shape[0]):
            yield idx, {"image": images[idx], "label": int(labels[idx])}


# For local development / debugger support only
if __name__ == "__main__":
    FashionMnistCorrupted().download_and_prepare()
