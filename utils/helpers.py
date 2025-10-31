"""
Utility functions for CNN CIFAR-10 project

FILE PURPOSE:
    Provides common utility functions used throughout the CNN CIFAR-10 project.
    Includes reproducibility setup, directory management, time formatting,
    experiment summarization, and various helper functions for data processing
    and visualization tasks.

DEPENDENCIES (IMPORTS FROM):
    - os: File system operations and directory management
    - json: JSON data serialization for configuration
    - numpy: Numerical computing for random seed setting
    - matplotlib.pyplot: Plotting utilities for visualization
    - datetime: Time and date handling
    - tensorflow: Deep learning framework for random seed setting

OUTPUTS (GENERATES):
    - Directory structure: Creates necessary project directories
    - Formatted output: Time formatting, experiment summaries
    - Reproducible results: Random seed configuration
    - Utility data: Various helper data structures and functions

ROLE IN PROJECT:
    Shared utility component that provides common functionality across modules.
    Used by main.py and other modules for directory setup, reproducibility,
    and various helper operations. Ensures consistent behavior and code reuse.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}")

def create_directories(base_path, subdirs):
    """
    Create directories if they don't exist
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectories to create
    """
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created/verified: {dir_path}")

def save_config(config_dict, filepath):
    """
    Save configuration dictionary to JSON file
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save the configuration
    """
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Configuration saved to {filepath}")

def load_config(filepath):
    """
    Load configuration from JSON file
    
    Args:
        filepath: Path to load the configuration from
    
    Returns:
        dict: Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config

def format_time(seconds):
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def plot_sample_images(images, labels, class_names, num_samples=10, 
                      figsize=(15, 6), save_plot=True, filepath=None):
    """
    Plot sample images with their labels
    
    Args:
        images: Array of images
        labels: Array of labels
        class_names: List of class names
        num_samples: Number of samples to display
        figsize: Figure size
        save_plot: Whether to save the plot
        filepath: Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: Sample images plot
    """
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.suptitle('Sample CIFAR-10 Images', fontsize=16)
    plt.tight_layout()
    
    if save_plot:
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'sample_images_{timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Sample images plot saved to {filepath}")
    
    plt.show()
    return fig

def print_model_info(model):
    """
    Print model information
    
    Args:
        model: Keras model
    """
    print("\nModel Information:")
    print("=" * 50)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"Non-trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    print("\nModel Architecture:")
    print("-" * 30)
    model.summary()

def calculate_model_size(model, filepath=None):
    """
    Calculate and save model size information
    
    Args:
        model: Keras model
        filepath: Path to save model size info
    
    Returns:
        dict: Model size information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    size_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'estimated_size_mb': model_size_mb
    }
    
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(size_info, f, indent=4)
        print(f"Model size info saved to {filepath}")
    
    return size_info

def create_experiment_log(experiment_name, config, results=None, filepath=None):
    """
    Create experiment log
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration used
        results: Results dictionary
        filepath: Path to save the log
    
    Returns:
        dict: Experiment log
    """
    log = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results or {}
    }
    
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(log, f, indent=4)
        print(f"Experiment log saved to {filepath}")
    
    return log

def compare_models(model_results, save_plot=True, filepath=None):
    """
    Compare multiple model results
    
    Args:
        model_results: Dictionary of model results
        save_plot: Whether to save the plot
        filepath: Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    models = list(model_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [model_results[model].get(metric, 0) for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_plot:
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'model_comparison_{timestamp}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {filepath}")
    
    plt.show()
    return fig

def print_experiment_summary(experiment_name, config, results, training_time=None):
    """
    Print experiment summary
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration used
        results: Results dictionary
        training_time: Training time
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {experiment_name}")
    print("=" * 60)
    
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if training_time:
        print(f"Training Time: {format_time(training_time)}")
    
    print("\nConfiguration:")
    print("-" * 20)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    if results:
        print("\nResults:")
        print("-" * 20)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    print("=" * 60)
