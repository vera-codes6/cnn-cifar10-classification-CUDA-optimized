"""
CNN Training Module - FIXED VERSION
===================================

This module handles CNN model training with proper callback configuration.
FIXED: Early stopping now monitors val_accuracy instead of val_loss to ensure
the most accurate model is restored.

FILE PURPOSE:
    Handles CNN model training with proper callbacks, GPU optimization,
    and ensures the best accuracy model is restored.

DEPENDENCIES (IMPORTS FROM):
    - tensorflow: Core deep learning framework
    - numpy: Numerical computing
    - matplotlib.pyplot: Plotting training history
    - os: File system operations
    - datetime: Time tracking
    - configs.config: PATHS_CONFIG for model saving

OUTPUTS (GENERATES):
    - Training history with loss and accuracy metrics
    - Model checkpoints saved to models/ directory
    - Training plots and visualizations
    - TensorBoard logs for monitoring

ROLE IN PROJECT:
    Core training component that trains CNN models with proper callback
    configuration to ensure the most accurate model is always restored.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from configs.config import PATHS_CONFIG

class CNNTrainer:
    """
    CNN Training class with FIXED callback configuration
    """
    
    def __init__(self, model, config=None):
        """
        Initialize trainer
        
        Args:
            model: Keras model to train
            config: Training configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.history = None
        self.training_start_time = None
        
    def train(self, train_data, validation_data=None, epochs=50, 
              batch_size=32, learning_rate=0.001, verbose=1):
        """
        Train the CNN model with FIXED callbacks
        
        Args:
            train_data: Training data (images, labels) or generator
            validation_data: Validation data or generator
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
        
        Returns:
            keras.callbacks.History: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        self.training_start_time = datetime.now()
        
        # Callbacks
        has_validation = validation_data is not None
        callbacks = self._get_callbacks(has_validation_data=has_validation)
        
        # Train the model
        if isinstance(train_data, tuple):
            # Direct data training
            train_images, train_labels = train_data
            if validation_data:
                val_images, val_labels = validation_data
                validation_data = (val_images, val_labels)
            
            self.history = self.model.fit(
                train_images, train_labels,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            # Generator training
            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose
            )
        
        training_time = datetime.now() - self.training_start_time
        print(f"Training completed in {training_time}")
        
        return self.history
    
    def _get_callbacks(self, has_validation_data=True):
        """
        Get training callbacks with FIXED configuration
        
        Args:
            has_validation_data: Whether validation data is available
        
        Returns:
            list: List of callbacks
        """
        callbacks = []
        
        # Using fixed learning rate - no scheduler needed
        # Learning rate is set when compiling the model
        
        # Early stopping and other validation-dependent callbacks
        if has_validation_data:
            # FIXED: Early stopping now monitors val_accuracy for best model
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # FIXED: Changed from val_loss to val_accuracy
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau (as backup)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # Keep this as val_loss for learning rate reduction
                factor=self.config.get('reduce_lr_factor', 0.3),
                patience=self.config.get('reduce_lr_patience', 8),
                min_lr=self.config.get('min_lr', 1e-6),
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Model checkpoint - monitors val_accuracy
            checkpoint_path = os.path.join(PATHS_CONFIG['models_dir'], 'best_model.keras')
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
        else:
            # Fallback callbacks when no validation data
            print("Warning: No validation data provided. Using training metrics for callbacks.")
            
            # Early stopping on training loss
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Model checkpoint on training accuracy
            checkpoint_path = os.path.join(PATHS_CONFIG['models_dir'], 'best_model.keras')
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # TensorBoard logging
        log_dir = os.path.join(PATHS_CONFIG['logs_dir'], 
                              f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def plot_training_history(self, save_plot=True):
        """
        Plot training history
        
        Args:
            save_plot: Whether to save the plot
        
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return None
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {plot_path}")
        
        return fig
    
    def get_best_epoch(self):
        """
        Get the epoch with the best validation accuracy
        
        Returns:
            int: Best epoch number
        """
        if self.history is None:
            return None
        
        history = self.history.history
        if 'val_accuracy' in history:
            best_epoch = np.argmax(history['val_accuracy']) + 1
            best_accuracy = max(history['val_accuracy'])
            print(f"Best epoch: {best_epoch} with validation accuracy: {best_accuracy:.4f}")
            return best_epoch
        else:
            best_epoch = np.argmax(history['accuracy']) + 1
            best_accuracy = max(history['accuracy'])
            print(f"Best epoch: {best_epoch} with training accuracy: {best_accuracy:.4f}")
            return best_epoch
