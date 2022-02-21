import os

from datasets import load_dataset

dataset = load_dataset("./fashion-mnist-c.py")

assert len(dataset) == 60000
# TODO Assertions