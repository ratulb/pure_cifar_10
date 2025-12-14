#!/usr/bin/env python
# test_script.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # Add current directory to path

import pure_cifar_10
import numpy as np

def test_download():
    """Test the download and extraction process."""
    print("=== Test 1: Download & Initialization ===")
    # Use a fresh directory to force download
    import shutil
    test_dir = "./fresh_test_data"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # This should trigger download
    dataset = pure_cifar_10.CIFAR10(folder=test_dir)
    print(f"✓ Dataset initialized in {test_dir}")

    return dataset

def test_data_shapes(dataset):
    """Verify data shapes and types."""
    print("\n=== Test 2: Data Shapes & Types ===")

    # Train data
    train_data = dataset.train_data
    train_labels = dataset.train_labels
    print(f"Train data: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"Train labels: {train_labels.shape}, dtype: {train_labels.dtype}")

    # Test data
    test_data = dataset.test_data
    test_labels = dataset.test_labels
    print(f"Test data: {test_data.shape}, dtype: {test_data.dtype}")
    print(f"Test labels: {test_labels.shape}, dtype: {test_labels.shape}")

    assert train_data.shape == (50000, 3, 32, 32), "Wrong train data shape"
    assert test_data.shape == (10000, 3, 32, 32), "Wrong test data shape"
    assert len(train_labels) == 50000, "Wrong train labels count"
    assert len(test_labels) == 10000, "Wrong test labels count"
    print("✓ All shapes correct!")

def test_data_values(dataset):
    """Verify data ranges and basic statistics."""
    print("\n=== Test 3: Data Value Ranges ===")

    train_data = dataset.train_data
    print(f"Min pixel value: {train_data.min():.3f}")
    print(f"Max pixel value: {train_data.max():.3f}")
    print(f"Mean pixel value: {train_data.mean():.3f}")
    print(f"Std pixel value: {train_data.std():.3f}")

    # CIFAR-10 should have pixel values in [0, 255] as uint8 in original
    # Our loader converts to float32 but keeps raw values
    assert 0 <= train_data.min() <= 255, "Pixel values out of expected range"
    assert 0 <= train_data.max() <= 255, "Pixel values out of expected range"
    print("✓ Pixel values in expected range!")

def test_class_names(dataset):
    """Verify class names are correct."""
    print("\n=== Test 4: Class Names ===")
    names = dataset.class_names
    print(f"Class names: {names}")
    assert len(names) == 10, "Should have 10 classes"
    assert names[0] == 'airplane', "First class should be airplane"
    assert names[-1] == 'truck', "Last class should be truck"
    print("✓ Class names correct!")

def test_load_function(dataset):
    """Test the convenience load() function."""
    print("\n=== Test 5: load() Function ===")
    train_data, train_labels, test_data, test_labels = dataset.load_all()

    print(f"load() returned: {len(train_data)} train, {len(test_data)} test")
    assert len(train_data) == 50000, "load() train data incorrect"
    assert len(test_data) == 10000, "load() test data incorrect"
    print("✓ load() function works!")

def main():
    print("Testing pure_cifar_10 module...")

    # Run tests
    dataset = test_download()
    test_data_shapes(dataset)
    test_data_values(dataset)
    test_class_names(dataset)
    test_load_function(dataset)

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)

    # Quick demo of usage
    print("\n--- Sample usage ---")
    print(f"Sample image shape: {dataset.train_data[0].shape}")
    print(f"Sample label: {dataset.train_labels[0]} -> {dataset.class_names[dataset.train_labels[0]]}")

if __name__ == "__main__":
    main()
