import os.path
import shutil
import unittest
from typing import List

import datasets
import numpy as np
from numpy.testing import assert_array_equal


def _img_array(pil_imgs: List) -> np.ndarray:
    return np.array([np.array(img) for img in pil_imgs])


def _label_array(labels: List[int]) -> np.ndarray:
    return np.array(labels)


class HuggingfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache_dir = "./huggingface-test-cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        ds = datasets.load_dataset("fashion_mnist_corrupted.py", cache_dir=cache_dir)
        cls.x_train = _img_array(ds["train"]["image"])
        cls.y_train = _label_array(ds["train"]["label"])
        cls.x_test = _img_array(ds["test"]["image"])
        cls.y_test = _label_array(ds["test"]["label"])

    def test_shapes(self):
        self.assertEqual(self.x_test.shape, (10000, 28, 28))
        self.assertEqual(self.y_test.shape, (10000,))
        self.assertEqual(self.x_train.shape, (60000, 28, 28))
        self.assertEqual(self.y_train.shape, (60000,))

    def test_label_counts(self):
        test_labls_counts = np.unique(self.y_test, return_counts=True)[1]
        assert_array_equal(
            test_labls_counts, np.full(shape=10, fill_value=1000, dtype=int)
        )

        train_labls_counts = np.unique(self.y_train, return_counts=True)[1]
        assert_array_equal(
            train_labls_counts, np.full(shape=10, fill_value=6000, dtype=int)
        )

    def test_same_as_local_npy(self):
        x_train_local = np.load("../generated/npy/fmnist-c-train.npy")
        assert_array_equal(self.x_train, x_train_local)
        y_train_local = np.load("../generated/npy/fmnist-c-train-labels.npy")
        assert_array_equal(self.y_train, y_train_local)
        x_test_local = np.load("../generated/npy/fmnist-c-test.npy")
        assert_array_equal(self.x_test, x_test_local)
        y_test_local = np.load("../generated/npy/fmnist-c-test-labels.npy")
        assert_array_equal(self.y_test, y_test_local)


if __name__ == "__main__":
    unittest.main()
