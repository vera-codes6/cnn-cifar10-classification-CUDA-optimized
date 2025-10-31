#!/usr/bin/env python3
"""
Best Model Training Script
==========================

This script reads the comprehensive hyperparameter tuning results, selects the best model,
and trains it for 50 epochs to get the final optimized performance.

Dependencies:
- pandas: For reading CSV results
- tensorflow: For model training
- src modules: For model creation and training
- utils modules: For GPU optimization

Outputs:
- Trained best model saved as 'final_best_model.keras'
- Performance metrics and comparison
- Training plots and evaluation results
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import warnings
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
tf.get_logger().setLevel('CRITICAL')

# Add project paths
sys.path.append('src')
sys.path.append('utils')

from models.cnn_model import CIFAR10CNN
from src.trainer import CNNTrainer
from src.evaluator import CNNEvaluator
from src.data_loader import load_cifar10_data, get_data_info
from utils.gpu_optimizer import setup_gpu_environment, get_optimal_batch_size, create_optimized_data_pipeline
from configs.config import PATHS_CONFIG

def load_best_model_settings():
    """Load the best model settings from comprehensive results"""
    results_file = os.path.join(PATHS_CONFIG['results_dir'], 'comprehensive_model_comparison.csv')
    
    if not os.path.exists(results_file):
        print("❌ Comprehensive results file not found!")
        print("Please run hyperparameter tuning first.")
        return None
    
    # Read results and get best model
    df = pd.read_csv(results_file)
    best_model = df.iloc[0]  # First row is the best (sorted by accuracy)
    
    print("🏆 BEST MODEL FOUND:")
    print(f"   Model: {best_model['Model']}")
    print(f"   Learning Rate: {best_model['Learning Rate']}")
    print(f"   Batch Size: {best_model['Batch Size']}")
    print(f"   Filters: {best_model['Filters']}")
    print(f"   Layers: {best_model['Layers']}")
    print(f"   Tuning Accuracy: {best_model['Accuracy']}")
    print(f"   Model File: {best_model['Model File']}")
    
    return {
        'learning_rate': float(best_model['Learning Rate']),
        'batch_size': int(best_model['Batch Size']),
        'num_filters': int(best_model['Filters']),
        'num_layers': int(best_model['Layers']),
        'model_file': best_model['Model File'],
        'tuning_accuracy': float(best_model['Accuracy'])
    }

def train_best_model(settings, train_data, test_data, class_names):
    """Train the best model for 50 epochs"""
    print(f"\n🚀 TRAINING BEST MODEL FOR 50 EPOCHS")
    print("=" * 50)
    
    # Create model with best settings
    cnn = CIFAR10CNN()
    model = cnn.build_model(
        num_filters=settings['num_filters'],
        num_layers=settings['num_layers']
    )
    model = cnn.compile_model(learning_rate=settings['learning_rate'])
    
    # GPU optimization
    optimal_batch_size = get_optimal_batch_size(
        model, train_data[0], max_batch_size=settings['batch_size'] * 4
    )
    
    # Create optimized data pipeline
    train_dataset = create_optimized_data_pipeline(
        tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1])),
        optimal_batch_size
    )
    val_dataset = create_optimized_data_pipeline(
        tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])),
        optimal_batch_size
    )
    
    # Create trainer
    trainer = CNNTrainer(model)
    
    print(f"📊 Training Settings:")
    print(f"   Learning Rate: {settings['learning_rate']}")
    print(f"   Optimal Batch Size: {optimal_batch_size}")
    print(f"   Filters: {settings['num_filters']}")
    print(f"   Layers: {settings['num_layers']}")
    print(f"   Epochs: 50")
    
    # Train the model
    start_time = datetime.now()
    history = trainer.train(
        train_data=train_dataset,
        validation_data=val_dataset,
        epochs=50,
        batch_size=optimal_batch_size,
        learning_rate=settings['learning_rate']
    )
    training_time = datetime.now() - start_time
    
    print(f"\n✅ Training completed in {training_time}")
    
    return model, history, optimal_batch_size

def evaluate_and_compare(model, test_data, class_names, settings, tuning_accuracy):
    """Evaluate the final model and compare with tuning results"""
    print(f"\n📊 EVALUATING FINAL MODEL")
    print("=" * 50)
    
    # Evaluate model
    evaluator = CNNEvaluator(model, class_names)
    metrics = evaluator.evaluate(test_data[0], test_data[1])
    
    # Create comparison table
    print(f"\n🏆 FINAL RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Tuning (20 epochs)':<20} {'Final (50 epochs)':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {tuning_accuracy:<20.4f} {metrics['accuracy']:<20.4f}")
    print(f"{'Precision':<20} {'N/A':<20} {metrics['precision_weighted']:<20.4f}")
    print(f"{'Recall':<20} {'N/A':<20} {metrics['recall_weighted']:<20.4f}")
    print(f"{'F1-Score':<20} {'N/A':<20} {metrics['f1_score_weighted']:<20.4f}")
    
    # Calculate improvement
    improvement = metrics['accuracy'] - tuning_accuracy
    print(f"\n📈 Improvement: {improvement:+.4f} ({improvement/tuning_accuracy*100:+.2f}%)")
    
    if improvement > 0:
        print("🎉 Model improved with longer training!")
    else:
        print("⚠️  Model may have overfitted or reached convergence")
    
    # Save final model
    final_model_path = os.path.join(PATHS_CONFIG['models_dir'], 'final_best_model.keras')
    model.save(final_model_path)
    print(f"\n💾 Final model saved to: {final_model_path}")
    
    # Create plots
    evaluator.plot_confusion_matrix()
    evaluator.create_metrics_table(metrics, "Final Best Model (50 epochs)")
    
    return metrics

def main():
    """Main execution function"""
    print("🚀 BEST MODEL TRAINING SCRIPT")
    print("=" * 50)
    
    # Setup GPU
    setup_gpu_environment()
    
    # Load best model settings
    settings = load_best_model_settings()
    if settings is None:
        return
    
    # Load data
    print(f"\n📊 Loading CIFAR-10 Dataset")
    print("-" * 30)
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10_data()
    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)
    print(f"✅ Data loaded: {train_data[0].shape[0]} train, {test_data[0].shape[0]} test samples")
    
    # Train best model
    model, history, optimal_batch_size = train_best_model(
        settings, train_data, test_data, class_names
    )
    
    # Evaluate and compare
    final_metrics = evaluate_and_compare(
        model, test_data, class_names, settings, settings['tuning_accuracy']
    )
    
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 50)
    print(f"Best Model Settings: LR={settings['learning_rate']}, BS={optimal_batch_size}, "
          f"Filters={settings['num_filters']}, Layers={settings['num_layers']}")
    print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final F1-Score: {final_metrics['f1_score_weighted']:.4f}")
    print(f"Model saved as: final_best_model.keras")

if __name__ == "__main__":
    main()
