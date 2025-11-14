"""
CNN CIFAR-10 Classification Project - GPU Optimized
High-performance deep learning with NVIDIA GPU acceleration

FILE PURPOSE:
    Main execution script that orchestrates the entire CNN CIFAR-10 classification pipeline.
    Handles command-line arguments, GPU setup, data loading, model training, evaluation,
    and hyperparameter tuning with comprehensive analysis and visualization.

DEPENDENCIES (IMPORTS FROM):
    - src.data_loader: load_cifar10_data, preprocess_data_for_training
    - models.cnn_model: CIFAR10CNN
    - src.trainer: CNNTrainer
    - src.evaluator: CNNEvaluator
    - src.feature_visualizer: FeatureVisualizer
    - src.hyperparameter_tuner: HyperparameterTuner
    - utils.helpers: set_random_seeds, create_directories, print_experiment_summary, format_time
    - utils.gpu_optimizer: setup_gpu_environment, print_gpu_info, get_optimal_batch_size, create_optimized_data_pipeline
    - configs.config: DATASET_CONFIG, TRAINING_CONFIG, HYPERPARAMETER_CONFIG, PATHS_CONFIG

OUTPUTS (GENERATES):
    - models/cnn_model.keras: Trained CNN model
    - models/optimized_cnn_model.keras: Hyperparameter-optimized model
    - plots/training_history.png: Training progress visualization
    - plots/confusion_matrix.png: Model performance confusion matrix
    - plots/feature_maps_*.png: Layer-wise feature visualizations
    - plots/class_metrics.png: Per-class performance metrics
    - plots/misclassifications.png: Error analysis visualization
    - results/classification_report.txt: Detailed performance report
    - results/performance_metrics.csv: Quantitative metrics
    - results/feature_analysis.txt: Feature layer analysis
    - logs/: TensorBoard training logs

ROLE IN PROJECT:
    Central orchestrator that coordinates all components of the deep learning pipeline.
    Manages GPU optimization, data flow, model lifecycle, and result generation.
    Provides command-line interface for different execution modes (train, tune, full).
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import warnings
import logging

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
tf.get_logger().setLevel('CRITICAL')

# Add src and utils to path
sys.path.append('src')
sys.path.append('utils')

# Import project modules
from src.data_loader import load_cifar10_data, preprocess_data_for_training
from models.cnn_model import CIFAR10CNN
from src.trainer import CNNTrainer
from src.evaluator import CNNEvaluator
from src.feature_visualizer import FeatureVisualizer
from src.hyperparameter_tuner import HyperparameterTuner
from utils.helpers import (set_random_seeds, create_directories, 
                          print_experiment_summary, format_time)
from utils.gpu_optimizer import (setup_gpu_environment, print_gpu_info, 
                                get_optimal_batch_size, create_optimized_data_pipeline)
from configs.config import (DATASET_CONFIG, TRAINING_CONFIG, 
                           HYPERPARAMETER_CONFIG, PATHS_CONFIG)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CNN CIFAR-10 Classification Project - GPU Optimized')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'tune'], 
                       default='full', help='Execution mode: full (default), train, or tune')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    return parser.parse_args()

def main():
    """Main execution function with GPU optimization"""
    args = parse_arguments()
    
    print("🚀 CNN CIFAR-10 Classification Project - GPU Optimized")
    print("=" * 60)
    print(f"Execution mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    # Setup GPU environment and optimizations
    print("\n🔧 Setting up GPU environment...")
    gpu_available = setup_gpu_environment()
    print_gpu_info()
    
    if not gpu_available:
        print("❌ GPU not available. Please check your NVIDIA drivers.")
        return
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Create necessary directories
    create_directories('.', ['models', 'results', 'plots', 'logs'])
    
    # Load and preprocess data
    print("\n📊 Loading and Preprocessing CIFAR-10 Dataset")
    print("-" * 50)
    start_time = time.time()
    
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10_data()
    train_images, train_labels, test_images, test_labels = preprocess_data_for_training(
        train_images, train_labels, test_images, test_labels
    )
    
    data_loading_time = time.time() - start_time
    print(f"✅ Data loading completed in {format_time(data_loading_time)}")
    print(f"Training samples: {len(train_images):,}")
    print(f"Test samples: {len(test_images):,}")
    
    # Initialize variables
    model = None
    training_time = 0
    metrics = None
    
    # Execute based on mode
    if args.mode in ['full', 'train']:
        # Build and train CNN model
        print("\n🏗️  Building and Training CNN Model")
        print("-" * 50)
        
        # Create model
        cnn = CIFAR10CNN()
        model = cnn.build_model()
        model = cnn.compile_model(learning_rate=args.learning_rate)
        
        print_model_info(model)
        
        # Determine optimal batch size
        print("\n⚡ Determining optimal batch size...")
        optimal_batch_size = get_optimal_batch_size(model, train_images)
        if args.batch_size:
            optimal_batch_size = args.batch_size
        print(f"Using batch size: {optimal_batch_size}")
        
        # Create optimized data pipeline
        print("🔧 Creating optimized data pipeline...")
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
        train_dataset = create_optimized_data_pipeline(train_dataset, optimal_batch_size)
        val_dataset = create_optimized_data_pipeline(val_dataset, optimal_batch_size)
        
        # Train model
        trainer = CNNTrainer(model, TRAINING_CONFIG)
        training_start = time.time()
        
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        history = trainer.train(
            train_data=train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            batch_size=optimal_batch_size,
            learning_rate=args.learning_rate
        )
        
        training_time = time.time() - training_start
        print(f"✅ Training completed in {format_time(training_time)}")
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model performance
        print("\n📈 Evaluating Model Performance")
        print("-" * 50)
        
        evaluator = CNNEvaluator(model, class_names)
        metrics = evaluator.evaluate(test_images, test_labels)
        
        # Generate evaluation plots and reports
        evaluator.plot_confusion_matrix()
        evaluator.plot_class_metrics(metrics)
        evaluator.generate_classification_report()
        evaluator.create_metrics_table(metrics, "CNN Model")
        evaluator.analyze_misclassifications(test_images)
        
        # Feature map visualization
        print("\n🔍 Feature Map Visualization")
        print("-" * 50)
        
        feature_visualizer = FeatureVisualizer(model, class_names)
        sample_images = test_images[:5]
        feature_visualizer.visualize_all_layers(sample_images, max_filters=8, max_images=2)
        feature_visualizer.analyze_feature_evolution(sample_images)
        
        # Get layer statistics
        statistics = feature_visualizer.get_layer_statistics(test_images[:100])
        feature_visualizer.print_layer_analysis(statistics)
        feature_visualizer.save_feature_analysis(statistics)
        
        # Save model
        model_path = os.path.join(PATHS_CONFIG['models_dir'], 'cnn_model.keras')
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")
        
        if args.mode == 'train':
            print("\n✅ Training mode completed successfully!")
            return
    
    # Hyperparameter tuning (for full and tune modes)
    if args.mode in ['full', 'tune']:
        print("\n🔧 Hyperparameter Tuning")
        print("-" * 50)
        
        tuner = HyperparameterTuner(
            train_data=(train_images, train_labels),
            test_data=(test_images, test_labels),
            class_names=class_names
        )
        
        # Run ablation study
        tuning_start = time.time()
        results_df = tuner.run_ablation_study()
        tuning_time = time.time() - tuning_start
        
        print(f"✅ Hyperparameter tuning completed in {format_time(tuning_time)}")
        
        # Plot hyperparameter analysis
        tuner.plot_hyperparameter_analysis(results_df)
        tuner.create_comparison_table(results_df)
        
        # Train best model with optimal hyperparameters for 50 epochs
        print("\n🏆 Training Best Model with Optimal Hyperparameters")
        print("-" * 50)
        
        # Get best parameters from tuning results
        best_params = tuner.best_params
        print(f"Best parameters found: {best_params}")
        
        # Create best model with optimal settings
        cnn = CIFAR10CNN()
        best_model = cnn.build_model(
            num_filters=best_params['num_filters'],
            num_layers=best_params['num_layers']
        )
        best_model = cnn.compile_model(learning_rate=best_params['learning_rate'])
        
        # Create trainer for best model
        best_trainer = CNNTrainer(best_model)
        
        # Train for 50 epochs with optimal settings
        print(f"🚀 Training best model for 50 epochs with optimal settings...")
        print(f"   Learning Rate: {best_params['learning_rate']}")
        print(f"   Batch Size: {best_params['batch_size']}")
        print(f"   Filters: {best_params['num_filters']}")
        print(f"   Layers: {best_params['num_layers']}")
        
        # Create optimized data pipeline
        optimal_batch_size = get_optimal_batch_size(best_model, train_images, 
                                                  initial_batch_size=best_params['batch_size'])
        train_dataset = create_optimized_data_pipeline(
            tf.data.Dataset.from_tensor_slices((train_images, train_labels)),
            optimal_batch_size
        )
        val_dataset = create_optimized_data_pipeline(
            tf.data.Dataset.from_tensor_slices((test_images, test_labels)),
            optimal_batch_size
        )
        
        # Train the best model
        best_history = best_trainer.train(
            train_data=train_dataset,
            validation_data=val_dataset,
            epochs=50,
            batch_size=optimal_batch_size,
            learning_rate=best_params['learning_rate']
        )
        
        # Evaluate best model
        best_evaluator = CNNEvaluator(best_model, class_names)
        best_metrics = best_evaluator.evaluate(test_images, test_labels)
        
        # Generate evaluation for best model
        best_evaluator.plot_confusion_matrix()
        best_evaluator.create_metrics_table(best_metrics, "Best Optimized CNN")
        
        # Save optimized model
        best_model_path = os.path.join(PATHS_CONFIG['models_dir'], 'best_optimized_cnn_model.keras')
        best_model.save(best_model_path)
        print(f"💾 Best optimized model saved to {best_model_path}")
        
        # Print final results
        print(f"\n🎯 FINAL BEST MODEL RESULTS:")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   Precision: {best_metrics['precision_weighted']:.4f}")
        print(f"   Recall: {best_metrics['recall_weighted']:.4f}")
        print(f"   F1-Score: {best_metrics['f1_weighted']:.4f}")
        
        if args.mode == 'tune':
            print("\n✅ Hyperparameter tuning mode completed successfully!")
            return
        
        # Final comparison (only for full mode)
        print("\n📊 Final Model Comparison")
        print("-" * 50)
        
        # Create comparison table
        comparison_data = {
            'Initial CNN': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision_weighted'],
                'recall': metrics['recall_weighted'],
                'f1_score': metrics['f1_weighted']
            },
            'Optimized CNN': {
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision_weighted'],
                'recall': best_metrics['recall_weighted'],
                'f1_score': best_metrics['f1_weighted']
            }
        }
        
        # Print comparison
        print("\nModel Performance Comparison:")
        print("=" * 60)
        print(f"{'Metric':<15} {'Initial CNN':<15} {'Optimized CNN':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            initial_val = comparison_data['Initial CNN'][metric]
            optimized_val = comparison_data['Optimized CNN'][metric]
            improvement = optimized_val - initial_val
            
            print(f"{metric.capitalize():<15} {initial_val:<15.4f} {optimized_val:<15.4f} {improvement:<15.4f}")
        
        # Print final summary
        print_experiment_summary(
            "CNN CIFAR-10 Complete Analysis - GPU Optimized",
            {
                'dataset': DATASET_CONFIG['name'],
                'total_samples': len(train_images) + len(test_images),
                'classes': len(class_names),
                'best_hyperparameters': tuner.best_params,
                'gpu_used': True,
                'optimal_batch_size': optimal_batch_size
            },
            {
                'initial_model': comparison_data['Initial CNN'],
                'optimized_model': comparison_data['Optimized CNN'],
                'training_time': training_time,
                'tuning_time': tuning_time,
                'total_time': time.time() - start_time
            },
            training_time + tuning_time
        )
    
    print("\n🎉 Project completed successfully!")
    print("All results, plots, and models have been saved to their respective directories.")

def print_model_info(model):
    """Print model information"""
    print("\n📋 Model Information:")
    print("=" * 60)
    print(f"Total parameters: {model.count_params():,}")
    
    print("\n🏗️  Model Architecture:")
    print("-" * 40)

    model.summary()

if __name__ == "__main__":
    main()