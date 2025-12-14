# pure_cifar_10

A pure Python, minimal-dependency loader for the CIFAR-10 dataset. This package provides the CIFAR-10 images and labels as NumPy arrays, with automatic downloading and caching, requiring only `numpy` and `tqdm`.

## Features
*   **Minimal Dependencies**: Only requires `numpy` and `tqdm`. No heavy machine learning frameworks like PyTorch or TensorFlow[citation:4].
*   **Automatic Handling**: Downloads the CIFAR-10 dataset automatically on first use and caches it locally.
*   **Pure NumPy**: Returns standard `numpy.ndarray` objects for easy integration into any pipeline.
*   **Progress Visualization**: Uses `tqdm` to show download and loading progress bars.
*   **Simple API**: Mirrors the clean, functional style of the `mnist_datasets` package.

## Installation
You can install the package directly from PyPI using pip:

```bash
pip install pure_cifar_10
```
## Usage
```python
from pure_cifar_10 import CIFAR10
loader = CIFAR10() 
train_data, train_labels, test_data, test_labels = loader.load_all()
```
