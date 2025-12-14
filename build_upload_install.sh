#!/usr/bin/env bash

sudo apt install python3.10-venv -y

# Uninstall old package
pip uninstall pure_cifar_10 -y

# Clean old builds
rm -rf dist/ *.egg-info/

# Build package
pip install --upgrade build
python3 -m build

# Install twine if not present
pip install --upgrade --user twine

# Upload using Python to avoid PATH issues
python3 -m twine upload dist/*

# Clear cache and reinstall
pip cache purge
sleep 2
pip install --no-cache-dir pure_cifar_10

