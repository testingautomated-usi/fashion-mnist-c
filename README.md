Note: As no changes to this codebase are planned, this repository is archived. 
If you have any questions, do not hesitate to contact us by email.

# FMNIST-C (Corrupted Fashion-Mnist)
[![Lint & Test](https://github.com/testingautomated-usi/fashion-mnist-c/actions/workflows/main.yml/badge.svg)](https://github.com/testingautomated-usi/fashion-mnist-c/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Docstr-Coverage](https://badgen.net/badge/docstr-coverage/100%25/green?cache=30)](https://github.com/HunterMcGushion/docstr_coverage)
[![Python Version](https://img.shields.io/pypi/pyversions/corrupted-text)](https://img.shields.io/pypi/pyversions/corrupted-text)
[![DOI](https://zenodo.org/badge/434567007.svg)](https://zenodo.org/badge/latestdoi/434567007)
[![License](https://badgen.net/badge/license/mit/blue?cache=30)](https://github.com/testingautomated-usi/fashion-mnist-c/blob/main/LICENSE)

This repository contains the source code used to create the FMNIST-C dataset, a
corrupted Fashion-MNIST benchmark for testing out-of-distribution robustness of computer
vision models.

[FMNIST](https://github.com/zalandoresearch/fashion-mnist) is a drop-in replacement for MNIST. FMNIST-C is a corresponding drop-in replacement for [MNIST-C](https://arxiv.org/abs/1906.02337).

## Corruptions
The following corruptions are applied to the images, equivalently to MNIST-C:

- **Noise** (shot noise and impulse noise)
- **Blur** (glass and motion blur)
- **Transformations** (shear, scale, rotate, brightness, contrast, saturate, inverse)

In addition, we apply various **image flippings and turnings**: For fashion images, flipping the image does not change its label,
and still keeps it a valid image. However, we noticed that in the nominal fmnist dataset, most images are identically oriented 
(e.g. most shoes point to the left side). Thus, flipped images provide valid OOD inputs.

Most corruptions are applied at a randomly selected level of *severity*, s.t. some corrupted images are really hard to classify whereas for others the corruption, while present, is subtle.

## Usage

The easiest way to use fashion-mnist-c is through huggingface datasets:

```python
# Install huggingface datasets
# pip install datasets

# The next two lines are all you need to load the corrupted dataset
from datasets import load_dataset
fmnist_c = load_dataset("mweiss/fashion_mnist_corrupted")

# Convert test sets numpy arrays (if you want)
#   You could of course do the same with the training set, but in most robustness studies, 
#   you'd use corrupted data only for testing, not for training.
import numpy as np
fmnist_c_x_test = np.array([np.array(x) for x in fmnist_c['test']['image']])
fmnist_c_y_test = np.array(fmnist_c['test']['label'])
```

Otherwise, this repository contains the binaries of the datasets in two formats:
- `./generated/npy/...` Numpy arrays.
- `./generated/ubyte/...` The file format used for the [original mnist dataset](http://yann.lecun.com/exdb/mnist/). These files can thus be used as drop-in replacements in most mnist dataset data loaders.

## Examples

| Turned  | Blurred | Rotated | Noise | Noise | Turned |
| ------------- | ------------- | --------| --------- | -------- | --------- |
| <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_0.png" width="100" height="100">   | <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_1.png" width="100" height="100"> |  <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_6.png" width="100" height="100"> |  <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_3.png" width="100" height="100"> |  <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_4.png" width="100" height="100"> |  <img src="https://raw.githubusercontent.com/testingautomated-usi/fashion-mnist-c/main/generated/png-examples/single_5.png" width="100" height="100"> |



## Citation
If you use this dataset, please cite the following paper:

```
@inproceedings{Weiss2022SimpleTechniques,
  title={Simple Techniques Work Surprisingly Well for Neural Network Test Prioritization and Active Learning},
  author={Weiss, Michael and Tonella, Paolo},
  booktitle={Proceedings of the 31th ACM SIGSOFT International Symposium on Software Testing and Analysis},
  year={2022}
}
```

Also, you may want to cite FMNIST and MNIST-C.

## Credits
- FMNIST-C is inspired by Googles MNIST-C and our repository is essentially a clone of theirs. See their [paper](https://arxiv.org/abs/1906.02337) and [repo](https://github.com/google-research/mnist-c).
- Find the nominal (i.e., non-corrupted) Fashion-MNIST dataset [here](https://github.com/zalandoresearch/fashion-mnist).

