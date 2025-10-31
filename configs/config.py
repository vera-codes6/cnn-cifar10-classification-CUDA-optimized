"""
Configuration file for CNN CIFAR-10 project
Contains all hyperparameters and settings

FILE PURPOSE:
    Central configuration file that defines all hyperparameters, settings, and paths
    used throughout the CNN CIFAR-10 project. Provides a single source of truth
    for model parameters, training settings, data configuration, and file paths.

DEPENDENCIES (IMPORTS FROM):
    - os: Operating system interface for path operations

OUTPUTS (GENERATES):
    - DATASET_CONFIG: CIFAR-10 dataset parameters and class names
    - MODEL_CONFIG: CNN architecture hyperparameters
    - TRAINING_CONFIG: Training process settings and callbacks
    - HYPERPARAMETER_CONFIG: Hyperparameter tuning search space
    - AUGMENTATION_CONFIG: Data augmentation parameters
    - PATHS_CONFIG: Directory paths for outputs and logs

ROLE IN PROJECT:
    Configuration management component that centralizes all project settings.
    Imported by all other modules to access consistent parameters and settings.
    Enables easy modification of hyperparameters without changing individual files.
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'uoft-cs/cifar10',
    'image_size': (32, 32),
    'num_classes': 10,
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'random_horizontal_flip': True,
    'random_crop_padding': 4,
    'normalize_mean': [0.4914, 0.4822, 0.4465],
    'normalize_std': [0.2023, 0.1994, 0.2010]
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'input_shape': (32, 32, 3),
    'num_classes': 10,
    'base_filters': 32,
    'num_conv_layers': 3,
    'dropout_rate': 0.5,
    'activation': 'relu',
    'final_activation': 'softmax'
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,  # Fixed learning rate - good for CIFAR-10
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],
    'validation_split': 0.2,
    'early_stopping_patience': 15,  # Increased patience
    'reduce_lr_patience': 8,  # Increased patience before reducing LR
    'reduce_lr_factor': 0.3,  # Less aggressive reduction (was 0.5)
    'min_lr': 1e-6  # Minimum learning rate
}

# Hyperparameter Tuning Configuration
HYPERPARAMETER_CONFIG = {
    'learning_rates': [0.0005, 0.001, 0.002],  # Good range for CIFAR-10 with fixed LR
    'batch_sizes': [16, 32, 64],
    'num_filters': [16, 32, 64],
    'num_layers': [3, 5, 7],
    'tuning_epochs': 20  # Reduced epochs for faster hyperparameter tuning
}

# Paths Configuration
PATHS_CONFIG = {
    'models_dir': 'models',
    'results_dir': 'results',
    'plots_dir': 'plots',
    'logs_dir': 'logs'
}

# Create directories if they don't exist
for dir_path in PATHS_CONFIG.values():
    os.makedirs(dir_path, exist_ok=True)
