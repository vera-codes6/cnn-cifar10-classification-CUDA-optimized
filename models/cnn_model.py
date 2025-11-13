"""
CNN Model Architecture for CIFAR-10 Classification

FILE PURPOSE:
    Defines the Convolutional Neural Network architecture for CIFAR-10 image classification.
    Provides flexible CNN model creation with configurable parameters, compilation,
    and utility functions for model management (save/load/summary).

DEPENDENCIES (IMPORTS FROM):
    - tensorflow: Core deep learning framework
    - tensorflow.keras: High-level neural network API
    - tensorflow.keras.layers: Neural network layer implementations
    - configs.config: MODEL_CONFIG for default model parameters

OUTPUTS (GENERATES):
    - keras.Model: Compiled CNN model ready for training
    - Model summaries: Architecture and parameter information
    - Saved models: .keras format model files

ROLE IN PROJECT:
    Core model definition component that creates the neural network architecture.
    Used by main.py for model instantiation and by trainer.py for training.
    Provides the foundation for all CNN operations in the classification pipeline.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from configs.config import MODEL_CONFIG

class CIFAR10CNN:
    """
    Convolutional Neural Network for CIFAR-10 classification
    """
    
    def __init__(self, config=None):
        """
        Initialize the CNN model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.model = None
        
    def build_model(self, num_filters=None, num_layers=None, dropout_rate=None):
        """
        Build the CNN model architecture
        
        Args:
            num_filters: Number of filters in first conv layer
            num_layers: Number of convolutional layers
            dropout_rate: Dropout rate for regularization
        
        Returns:
            keras.Model: Compiled CNN model
        """
        # Use provided parameters or defaults
        filters = num_filters or self.config['base_filters']
        conv_layers = num_layers or self.config['num_conv_layers']
        dropout = dropout_rate or self.config['dropout_rate']
        
        # Input layer
        inputs = keras.Input(shape=self.config['input_shape'])
        x = inputs
        
        # Convolutional layers
        for i in range(conv_layers):
            # Double filters after each pooling layer
            current_filters = filters * (2 ** (i // 2))
            
            x = layers.Conv2D(
                current_filters, 
                kernel_size=3, 
                activation=self.config['activation'],
                padding='same'
            )(x)
            x = layers.BatchNormalization()(x)
            
            # Add pooling every 2 layers
            if (i + 1) % 2 == 0:
                x = layers.MaxPooling2D(pool_size=2)(x)
                x = layers.Dropout(dropout * 0.5)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation=self.config['activation'])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Dense(64, activation=self.config['activation'])(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.config['num_classes'], 
            activation=self.config['final_activation']
        )(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='CIFAR10_CNN')
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name or instance
        
        Returns:
            keras.Model: Compiled model
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Create optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get model summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            # Try with legacy .h5 format if .keras fails
            if filepath.endswith('.keras'):
                legacy_path = filepath.replace('.keras', '.h5')
                try:
                    self.model = keras.models.load_model(legacy_path)
                    print(f"Model loaded from legacy format: {legacy_path}")
                except Exception as e2:
                    print(f"Error loading legacy model: {e2}")
                    raise e2
            else:
                raise e
        return self.model

def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a simple CNN model for quick testing
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
    
    Returns:
        keras.Model: Simple CNN model
    """
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_deep_cnn(input_shape=(32, 32, 3), num_classes=10, num_layers=7):
    """
    Create a deeper CNN model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        num_layers: Number of convolutional layers
    
    Returns:
        keras.Model: Deep CNN model
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    filters = 32
    for i in range(num_layers):
        model.add(layers.Conv2D(filters, 3, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        
        if (i + 1) % 2 == 0:
            model.add(layers.MaxPooling2D(2))
            model.add(layers.Dropout(0.25))
            filters *= 2
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
