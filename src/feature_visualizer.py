"""
Feature map visualization module for CNN layers

FILE PURPOSE:
    Visualizes and analyzes feature maps from CNN layers to understand model behavior.
    Extracts intermediate layer activations, creates feature map visualizations,
    analyzes feature evolution across layers, and provides statistical analysis
    of layer activations for model interpretability.

DEPENDENCIES (IMPORTS FROM):
    - numpy: Numerical computing for data manipulation
    - matplotlib.pyplot: Feature map visualization plotting
    - tensorflow: Deep learning framework for model operations
    - tensorflow.keras: High-level neural network API
    - os: File system operations
    - configs.config: PATHS_CONFIG, DATASET_CONFIG for paths and class names

OUTPUTS (GENERATES):
    - plots/feature_maps_conv_*.png: Layer-wise feature map visualizations
    - plots/feature_evolution.png: Feature evolution across layers
    - results/feature_analysis.txt: Statistical analysis of layer activations
    - Layer statistics: Mean, std, sparsity, activation patterns

ROLE IN PROJECT:
    Model interpretability component that provides insights into CNN behavior.
    Used by main.py to analyze how the model processes images and learns features.
    Enables understanding of model decision-making process and layer contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from configs.config import PATHS_CONFIG, DATASET_CONFIG

class FeatureVisualizer:
    """
    Class for visualizing feature maps from CNN layers
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize feature visualizer
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or DATASET_CONFIG['class_names']
        self.feature_extractors = {}
        
    def create_feature_extractors(self):
        """
        Create feature extractors for different layers
        """
        # Get all convolutional layers
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Convolution2D)):
                conv_layers.append(layer)
        
        print(f"Found {len(conv_layers)} convolutional layers")
        
        # Create feature extractors for each conv layer
        for i, layer in enumerate(conv_layers):
            feature_extractor = keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
            self.feature_extractors[f'conv_{i+1}'] = {
                'extractor': feature_extractor,
                'layer_name': layer.name,
                'layer': layer
            }
        
        return self.feature_extractors
    
    def extract_feature_maps(self, images, layer_name=None, max_images=5):
        """
        Extract feature maps for given images
        
        Args:
            images: Input images
            layer_name: Name of the layer to extract features from
            max_images: Maximum number of images to process
        
        Returns:
            dict: Feature maps for each layer
        """
        if not self.feature_extractors:
            self.create_feature_extractors()
        
        # Limit number of images
        images = images[:max_images]
        
        feature_maps = {}
        
        if layer_name:
            # Extract from specific layer
            if layer_name in self.feature_extractors:
                extractor = self.feature_extractors[layer_name]['extractor']
                # Use tf.function to avoid retracing warnings
                @tf.function(reduce_retracing=True)
                def predict_features(x):
                    return extractor(x, training=False)
                features = predict_features(images)
                feature_maps[layer_name] = features
            else:
                print(f"Layer {layer_name} not found in feature extractors")
        else:
            # Extract from all layers
            for name, extractor_info in self.feature_extractors.items():
                extractor = extractor_info['extractor']
                # Use tf.function to avoid retracing warnings
                @tf.function(reduce_retracing=True)
                def predict_features(x):
                    return extractor(x, training=False)
                features = predict_features(images)
                feature_maps[name] = features
        
        return feature_maps
    
    def visualize_feature_maps(self, images, layer_name, max_filters=16, 
                             max_images=3, save_plot=True):
        """
        Visualize feature maps for a specific layer
        
        Args:
            images: Input images
            layer_name: Name of the layer to visualize
            max_filters: Maximum number of filters to show
            max_images: Maximum number of images to process
            save_plot: Whether to save the plot
        
        Returns:
            matplotlib.figure.Figure: Feature maps visualization
        """
        if not self.feature_extractors:
            self.create_feature_extractors()
        
        if layer_name not in self.feature_extractors:
            print(f"Layer {layer_name} not found")
            return None
        
        # Extract feature maps
        feature_maps = self.extract_feature_maps(images, layer_name, max_images)
        features = feature_maps[layer_name]
        
        # Get layer info
        layer_info = self.feature_extractors[layer_name]
        num_filters = features.shape[-1]
        num_filters_to_show = min(max_filters, num_filters)
        
        # Create subplot
        fig, axes = plt.subplots(max_images, num_filters_to_show + 1, 
                               figsize=(num_filters_to_show * 2, max_images * 2))
        
        if max_images == 1:
            axes = axes.reshape(1, -1)
        
        for img_idx in range(max_images):
            # Original image - normalize to [0, 1] range
            original_img = images[img_idx]
            if original_img.min() < 0 or original_img.max() > 1:
                original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            axes[img_idx, 0].imshow(original_img)
            axes[img_idx, 0].set_title('Original')
            axes[img_idx, 0].axis('off')
            
            # Feature maps - normalize to [0, 1] range to avoid clipping warnings
            for filter_idx in range(num_filters_to_show):
                feature_map = features[img_idx, :, :, filter_idx]
                
                # Convert to numpy and normalize feature map to [0, 1] range
                feature_map = np.array(feature_map)
                if feature_map.min() != feature_map.max():
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                else:
                    feature_map = np.zeros_like(feature_map)
                
                axes[img_idx, filter_idx + 1].imshow(feature_map, cmap='viridis', vmin=0, vmax=1)
                axes[img_idx, filter_idx + 1].set_title(f'Filter {filter_idx + 1}')
                axes[img_idx, filter_idx + 1].axis('off')
        
        plt.suptitle(f'Feature Maps - {layer_info["layer_name"]}', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 
                                   f'feature_maps_{layer_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps saved to {plot_path}")
        
        plt.show()
        return fig
    
    def visualize_all_layers(self, images, max_filters=8, max_images=2, save_plot=True):
        """
        Visualize feature maps for all convolutional layers
        
        Args:
            images: Input images
            max_filters: Maximum number of filters to show per layer
            max_images: Maximum number of images to process
            save_plot: Whether to save the plot
        
        Returns:
            list: List of matplotlib figures
        """
        if not self.feature_extractors:
            self.create_feature_extractors()
        
        figures = []
        
        for layer_name in self.feature_extractors.keys():
            print(f"Visualizing layer: {layer_name}")
            fig = self.visualize_feature_maps(
                images, layer_name, max_filters, max_images, save_plot
            )
            if fig:
                figures.append(fig)
        
        return figures
    
    def analyze_feature_evolution(self, images, class_names_to_analyze=None, 
                                max_images_per_class=2, save_plot=True):
        """
        Analyze how features evolve through layers for different classes
        
        Args:
            images: Input images
            class_names_to_analyze: List of class names to analyze
            max_images_per_class: Maximum images per class
            save_plot: Whether to save the plot
        
        Returns:
            matplotlib.figure.Figure: Feature evolution analysis
        """
        if not self.feature_extractors:
            self.create_feature_extractors()
        
        # Select images from different classes
        if class_names_to_analyze is None:
            class_names_to_analyze = self.class_names[:3]  # Analyze first 3 classes
        
        # Create a large subplot for all layers and classes
        num_layers = len(self.feature_extractors)
        num_classes = len(class_names_to_analyze)
        
        fig, axes = plt.subplots(num_classes, num_layers, 
                               figsize=(num_layers * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        if num_layers == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, class_name in enumerate(class_names_to_analyze):
            # Find an image of this class (simplified - in practice, you'd have labels)
            # For now, we'll just use the first few images
            img_idx = class_idx * max_images_per_class
            
            for layer_idx, (layer_name, layer_info) in enumerate(self.feature_extractors.items()):
                # Extract features for this image and layer
                features = self.extract_feature_maps(images[img_idx:img_idx+1], layer_name)
                feature_map = features[layer_name][0, :, :, 0]  # First filter
                
                axes[class_idx, layer_idx].imshow(feature_map, cmap='viridis')
                axes[class_idx, layer_idx].set_title(f'{class_name}\n{layer_name}')
                axes[class_idx, layer_idx].axis('off')
        
        plt.suptitle('Feature Evolution Through Layers', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 'feature_evolution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Feature evolution analysis saved to {plot_path}")
        
        plt.show()
        return fig
    
    def get_layer_statistics(self, images, max_images=100):
        """
        Get statistics about feature maps for each layer
        
        Args:
            images: Input images
            max_images: Maximum number of images to analyze
        
        Returns:
            dict: Layer statistics
        """
        if not self.feature_extractors:
            self.create_feature_extractors()
        
        images = images[:max_images]
        statistics = {}
        
        for layer_name, layer_info in self.feature_extractors.items():
            features = self.extract_feature_maps(images, layer_name)
            feature_maps = features[layer_name]
            
            stats = {
                'layer_name': layer_info['layer_name'],
                'output_shape': feature_maps.shape,
                'num_filters': feature_maps.shape[-1],
                'mean_activation': np.mean(feature_maps),
                'std_activation': np.std(feature_maps),
                'max_activation': np.max(feature_maps),
                'min_activation': np.min(feature_maps),
                'sparsity': np.mean(feature_maps == 0)  # Percentage of zero activations
            }
            
            statistics[layer_name] = stats
        
        return statistics
    
    def print_layer_analysis(self, statistics):
        """
        Print analysis of layer statistics
        
        Args:
            statistics: Layer statistics dictionary
        """
        print("\nLayer Analysis:")
        print("=" * 80)
        
        for layer_name, stats in statistics.items():
            print(f"\nLayer: {stats['layer_name']}")
            print(f"  Output Shape: {stats['output_shape']}")
            print(f"  Number of Filters: {stats['num_filters']}")
            print(f"  Mean Activation: {stats['mean_activation']:.4f}")
            print(f"  Std Activation: {stats['std_activation']:.4f}")
            print(f"  Max Activation: {stats['max_activation']:.4f}")
            print(f"  Min Activation: {stats['min_activation']:.4f}")
            print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    def save_feature_analysis(self, statistics, filename=None):
        """
        Save feature analysis to file
        
        Args:
            statistics: Layer statistics dictionary
            filename: Custom filename
        """
        if filename is None:
            filename = 'feature_analysis.txt'
        
        filepath = os.path.join(PATHS_CONFIG['results_dir'], filename)
        
        with open(filepath, 'w') as f:
            f.write("CNN Feature Map Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for layer_name, stats in statistics.items():
                f.write(f"Layer: {stats['layer_name']}\n")
                f.write(f"  Output Shape: {stats['output_shape']}\n")
                f.write(f"  Number of Filters: {stats['num_filters']}\n")
                f.write(f"  Mean Activation: {stats['mean_activation']:.4f}\n")
                f.write(f"  Std Activation: {stats['std_activation']:.4f}\n")
                f.write(f"  Max Activation: {stats['max_activation']:.4f}\n")
                f.write(f"  Min Activation: {stats['min_activation']:.4f}\n")
                f.write(f"  Sparsity: {stats['sparsity']:.2%}\n\n")
        
        print(f"Feature analysis saved to {filepath}")
