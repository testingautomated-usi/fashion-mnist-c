# FMNIST-C (Corrupted Fashion-Mnist)

This repository contains the source code used to create the FMNIST-C dataset, a
corrupted Fashion-MNIST benchmark for testing out-of-distribution robustness of computer
vision models.

[FMNIST](https://github.com/zalandoresearch/fashion-mnist) is a drop-in replacement for MNIST and FMNIST-C is a corresponding drop-in replacement for [MNIST-C](https://arxiv.org/abs/1906.02337).

## Corruptions
The following corruptions are applied to the images, equivalently to MNIST-C:

- **Noise** (shot noise and impulse noise)
- **Blur** (glass and motion blurr)
- **Transformations** (shear, scale, rotate, brightness, constrast, saturate, inverse)

In addition, we apply various **image flippings**: For fashion images, flipping the image does not change it's label,
and still keeps it a valid image. However, we noticed that in the nominal fmnist dataset, most images are identically oriented 
(e.g. most shoes point to the left side). Thus, flipped images provide valid OOD inputs.

Most corruptions are applied at a randomly selected level of *severity*, s.t. some corrupted images are really hard to classify where for others the corruption, while present, is subtle.

## Usage

The easiest way to use fmnist-c is through huggingface datasets:
1. `pip install datasets`
3. **OMITTED** *(Huggingface dataset and corresponding instructions will be added here after the (double-blind!) review.)*

Otherwise, this repository contains the binaries of the datasets in two formats:
- `./generated/npy/...` Numpy arrays, to be loaded e.g. using `numpy.load(./generated/npy/fmnist-c-test.npy)`
- `./generated/ubyte/...` The file format used for the [original mnist dataset](http://yann.lecun.com/exdb/mnist/). These files can thus be used as drop-in replacements in most mnist dataset data loaders.

## Examples



## Citation
If you use this dataset, please cite the following paper:

[//]: # (TODO De-Anonymize)

```
Simple Techniques Work Surprisingly Well for Neural Network Test Prioritization and Active Learning, Under Review,
(authors anonymized)
```

## Credits
- FMNIST-C is inspired by Googles MNIST-C and our repository is essentially a clone of theirs. See their [paper](https://arxiv.org/abs/1906.02337) and [repo](https://github.com/google-research/mnist-c).
- Find the nominal (i.e., non-corrupted) Fashion-MNIST dataset [here](https://github.com/zalandoresearch/fashion-mnist).

