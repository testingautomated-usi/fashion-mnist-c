"""Utility to convert the numpy arrays into the mnist ubyte format."""

import os

import numpy as np


def npy_to_ubyte():
    """Converts the .npy files to fashion-mnist-like ubyte files"""
    for dirpath, dnames, fnames in os.walk("./generated/npy"):
        for f in fnames:
            if f.endswith(".npy"):
                a = np.load(os.path.join(dirpath, f), allow_pickle=True)
                # Same as file.write(a.tobytes())  (see numpy docs)
                filename = os.path.join(
                    "./generated/ubyte", f.replace(".npy", "-ubyte.gz")
                )
                if "label" in filename:
                    _write_labeldata(a, filename)
                else:
                    _write_imagedata(a, filename)

                filename = os.path.join(
                    "./generated/dummy", f.replace(".npy", "-ubyte.gz")
                )
                if "label" in filename:
                    _write_labeldata(a[:100], filename)
                else:
                    _write_imagedata(a[:100], filename)


# https://github.com/davidflanagan/notMNIST-to-MNIST/blob/17823f4d4a3acd8317c07866702d2eb2ac79c7a0/convert_to_mnist_format.py#L92
def _write_labeldata(labeldata, outputfile):
    labeldata = np.array(labeldata, dtype=np.uint8)
    header = np.array([0x0801, len(labeldata)], dtype=">i4")
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())


# https://github.com/davidflanagan/notMNIST-to-MNIST/blob/17823f4d4a3acd8317c07866702d2eb2ac79c7a0/convert_to_mnist_format.py#L92
def _write_imagedata(imagedata, outputfile):
    imagedata = np.array(imagedata, dtype=np.uint8)
    header = np.array([0x0803, len(imagedata), 28, 28], dtype=">i4")
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())


if __name__ == "__main__":
    npy_to_ubyte()
