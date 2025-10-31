"""
Data loading and preprocessing utilities for CIFAR-10 dataset
Modified to work with TensorFlow/Keras

FILE PURPOSE:
    Handles CIFAR-10 dataset loading, preprocessing, and augmentation for CNN training.
    Downloads data from Hugging Face, applies normalization, data augmentation,
    and converts data to TensorFlow-compatible formats for GPU-optimized training.

DEPENDENCIES (IMPORTS FROM):
    - tensorflow: Core deep learning framework
    - datasets: Hugging Face datasets library for CIFAR-10 access
    - numpy: Numerical computing for data manipulation
    - configs.config: DATASET_CONFIG, AUGMENTATION_CONFIG for parameters

OUTPUTS (GENERATES):
    - train_images, train_labels: Preprocessed training data (numpy arrays)
    - test_images, test_labels: Preprocessed test data (numpy arrays)
    - class_names: List of CIFAR-10 class names
    - Augmented datasets: Enhanced training data with transformations

ROLE IN PROJECT:
    Data pipeline component that prepares raw CIFAR-10 data for model consumption.
    Used by main.py to load and preprocess data before training.
    Ensures data is properly formatted and optimized for GPU training.
"""

import tensorflow as tf
from datasets import load_dataset
import numpy as np
from configs.config import DATASET_CONFIG, AUGMENTATION_CONFIG

def load_cifar10_data():
    """
    Load CIFAR-10 dataset from Hugging Face and convert to TensorFlow format
    
    Returns:
        tuple: (train_dataset, test_dataset, class_names)
    """
    print("Loading CIFAR-10 dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset(DATASET_CONFIG['name'])
        print("Dataset loaded successfully!")
        
        # Extract train and test splits
        train_data = dataset['train']
        test_data = dataset['test']
        
        # Convert to numpy arrays
        train_images = np.array([np.array(img) for img in train_data['img']])
        train_labels = np.array(train_data['label'])
        
        test_images = np.array([np.array(img) for img in test_data['img']])
        test_labels = np.array(test_data['label'])
        
        # Normalize pixel values to [0, 1]
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        print(f"Training data shape: {train_images.shape}")
        print(f"Test data shape: {test_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        
        return train_images, train_labels, test_images, test_labels, DATASET_CONFIG['class_names']
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_data_generators(train_images, train_labels, test_images, test_labels, 
                          batch_size=32, validation_split=0.2):
    """
    Create TensorFlow data generators with augmentation
    
    Args:
        train_images: Training images
        train_labels: Training labels
        test_images: Test images
        test_labels: Test labels
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
    
    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=AUGMENTATION_CONFIG['random_horizontal_flip'],
        zoom_range=0.1,
        validation_split=validation_split
    )
    
    # No augmentation for validation and test
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=batch_size,
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=batch_size,
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow(
        test_images, test_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def preprocess_data_for_training(train_images, train_labels, test_images, test_labels):
    """
    Preprocess data for direct training (without generators)
    
    Args:
        train_images: Training images
        train_labels: Training labels
        test_images: Test images
        test_labels: Test labels
    
    Returns:
        tuple: Preprocessed data ready for training
    """
    
    # Normalize using CIFAR-10 statistics
    mean = np.array(AUGMENTATION_CONFIG['normalize_mean'])
    std = np.array(AUGMENTATION_CONFIG['normalize_std'])
    
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    
    return train_images, train_labels, test_images, test_labels

def get_data_info():
    """
    Get dataset information
    
    Returns:
        dict: Dataset information
    """
    return {
        'num_classes': DATASET_CONFIG['num_classes'],
        'class_names': DATASET_CONFIG['class_names'],
        'image_size': DATASET_CONFIG['image_size'],
        'total_samples': 60000,
        'train_samples': 50000,
        'test_samples': 10000
    }
