# pure_cifar_10 — Agent Guide

## Quick start
```bash
python -c "from pure_cifar_10 import CIFAR10; d = CIFAR10(); d.load_all()"
```

## No test framework
Tests live in `test_script.py` — run with:
```bash
python test_script.py
```
`test_script.py` uses `sys.path.insert(0, ...)` to import the source package directly (no `pip install -e .` needed).

## Release workflow
- **Bump version:** `./bump_version.sh` — auto-increments `version="..."` in `setup.py` (patch for X.Y.Z, minor for X.Y)
- **Build + publish + verify:** `./build_upload_install.sh` — runs `python -m build`, `twine upload dist/*`, then `pip install --no-cache-dir pure_cifar_10`

## Package structure
```
pure_cifar_10/
  __init__.py   # exports CIFAR10
  loader.py     # CIFAR10 class — download, extract, load (channels-first, float32, [0,255])
```

## Dependencies
- runtime: `numpy>=2.0.0`, `tqdm>=4.66.0`
- Python >= 3.10
- No dev / lint / typecheck tooling, no CI, no Makefile

## Download cache
Default cache path: `/tmp/cifar10_data` (configurable via `folder=` kwarg).
