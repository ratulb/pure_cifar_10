---
layout: page
title: CIFAR-10 Dataset Loader
---

**A pure Python, minimal-dependency loader for the CIFAR-10 dataset.**

Pure Python + NumPy — no PyTorch, no TensorFlow, no framework lock-in. Download, extract, and load CIFAR-10 images and labels with a single class.

---

## Features

- **Minimal Dependencies** — only requires `numpy` and `tqdm`.
- **Automatic Handling** — downloads the CIFAR-10 dataset automatically on first use and caches it locally.
- **Pure NumPy** — returns standard `numpy.ndarray` objects for easy integration into any pipeline.
- **Progress Visualization** — uses `tqdm` to show download and loading progress bars.
- **Simple API** — mirrors the clean, functional style of the `mnist_datasets` package.

---

## Installation

```bash
pip install pure_cifar_10
```

Requires Python >= 3.10.

---

## Quick start

```python
from pure_cifar_10 import CIFAR10

loader = CIFAR10()
train_data, train_labels, test_data, test_labels = loader.load_all()
# train_data: (50000, 3, 32, 32) float32, train_labels: (50000,) int64
# test_data:  (10000, 3, 32, 32) float32, test_labels:  (10000,) int64
```

---

## API

### CIFAR10

| Method / Property | Returns | Description |
|-------------------|---------|-------------|
| `load_all()` | `(train_data, train_labels, test_data, test_labels)` | Load the complete dataset |
| `load(train=True)` | `(data, labels)` | Load train (`True`) or test (`False`) set |
| `train_data` | `np.ndarray` | Property — (50000, 3, 32, 32) float32 |
| `train_labels` | `np.ndarray` | Property — (50000,) int64 |
| `test_data` | `np.ndarray` | Property — (10000, 3, 32, 32) float32 |
| `test_labels` | `np.ndarray` | Property — (10000,) int64 |
| `class_names` | `tuple[str, ...]` | Property — 10 class name strings |

Data is lazily loaded — the first access to any `train_*` / `test_*` property triggers download, extraction, and parsing.

### Constructor

```python
CIFAR10(
    folder="/tmp/cifar10_data",   # Cache directory for downloads
    show_progress=True,            # Show tqdm progress bars
)
```

---

## Examples

### Custom cache directory

```python
loader = CIFAR10(folder="/tmp/data")
```

### Load train and test separately

```python
train_data, train_labels = loader.load(train=True)
test_data, test_labels = loader.load(train=False)
```

### Inspect data

```python
loader = CIFAR10()
train_data, train_labels, test_data, test_labels = loader.load_all()

print(train_data.shape)   # (50000, 3, 32, 32)
print(train_labels.shape) # (50000,)

classes = loader.class_names
print(classes)
# ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

---

## Data Format

| Attribute | Shape | Dtype | Range |
|-----------|-------|-------|-------|
| Images | (N, 3, 32, 32) | `float32` | [0, 255] |
| Labels | (N,) | `int64` | 0–9 |

- **Channels-first:** 3 color channels (Red, Green, Blue) — each a 32×32 grid.
- **Raw pixel values:** stored in the original [0, 255] range as float32 (not normalized).
- **Labels:** integers 0–9, one per image.

---

## Class Names

| Index | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## How It Works

1. **Download** — fetches `cifar-10-python.tar.gz` (~163 MB) from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` with a `tqdm` progress bar.
2. **Extract** — unpacks into `cifar-10-batches-py/` containing 5 training batches + 1 test batch (pickled dicts).
3. **Load** — each batch is deserialized with `pickle`, data is cast to `float32`, labels to `int64`, and batches are concatenated and reshaped to channels-first format.
4. **Cache** — extracted files are kept in `folder`; subsequent loads skip download/extract.

The tar archive is deleted after extraction to save disk space.

---

## Project structure

| Path | Purpose |
|------|---------|
| `pure_cifar_10/loader.py` | `CIFAR10` class |
| `pure_cifar_10/__init__.py` | Public API exports (`CIFAR10`) |
| `test_script.py` | Standalone test script (no test framework) |
| `setup.py` | PyPI packaging |

---

## License

MIT.
