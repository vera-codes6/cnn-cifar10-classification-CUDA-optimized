"""
Hyperparameter tuning module for CNN CIFAR-10

FILE PURPOSE:
    Performs systematic hyperparameter optimization through ablation studies.
    Tests different combinations of model parameters (filters, layers, dropout, etc.),
    trains multiple model variants, evaluates performance, and identifies optimal
    configurations for CIFAR-10 classification.

DEPENDENCIES (IMPORTS FROM):
    - itertools: Parameter combination generation
    - numpy: Numerical computing for data manipulation
    - pandas: Results data management and analysis
    - matplotlib.pyplot: Performance visualization plotting
    - seaborn: Statistical data visualization
    - datetime: Timestamp generation
    - os: File system operations
    - configs.config: HYPERPARAMETER_CONFIG, PATHS_CONFIG for parameters
    - models.cnn_model: CIFAR10CNN for model creation
    - src.trainer: CNNTrainer for model training
    - src.evaluator: CNNEvaluator for performance assessment

OUTPUTS (GENERATES):
    - Hyperparameter results: pandas.DataFrame with performance metrics
    - plots/hyperparameter_analysis.png: Performance comparison visualizations
    - results/hyperparameter_comparison.csv: Detailed results table
    - Best model: Optimized CNN model with best parameters
    - Performance analysis: Statistical comparison of different configurations

ROLE IN PROJECT:
    Model optimization component that finds the best hyperparameter configuration.
    Used by main.py to perform ablation studies and identify optimal settings.
    Provides systematic approach to model improvement and performance enhancement.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import tensorflow as tf
from tqdm import tqdm
from configs.config import HYPERPARAMETER_CONFIG, PATHS_CONFIG
from models.cnn_model import CIFAR10CNN
from src.trainer import CNNTrainer
from src.evaluator import CNNEvaluator
from utils.gpu_optimizer import get_optimal_batch_size, create_optimized_data_pipeline

class HyperparameterTuner:
    """
    Hyperparameter tuning class for CNN model
    """
    
    def __init__(self, train_data, test_data, class_names):
        """
        Initialize hyperparameter tuner
        
        Args:
            train_data: Training data (images, labels)
            test_data: Test data (images, labels)
            class_names: List of class names
        """
        self.train_data = train_data
        self.test_data = test_data
        self.class_names = class_names
        self.results = []
        self.best_params = {}
        self.best_score = 0.0
        
    def run_ablation_study(self, config=None):
        """
        Run ablation study for hyperparameter tuning
        
        Args:
            config: Hyperparameter configuration
        
        Returns:
            pandas.DataFrame: Results of ablation study
        """
        config = config or HYPERPARAMETER_CONFIG
        
        print("Starting Hyperparameter Ablation Study...")
        print("=" * 50)
        
        # Generate all combinations
        param_combinations = list(itertools.product(
            config['learning_rates'],
            config['batch_sizes'],
            config['num_filters'],
            config['num_layers']
        ))
        
        total_combinations = len(param_combinations)
        print(f"Total combinations to test: {total_combinations}")
        
        # Create progress bar
        pbar = tqdm(enumerate(param_combinations), total=total_combinations, 
                   desc="🔧 Hyperparameter Tuning", unit="model")
        
        for i, (lr, batch_size, num_filters, num_layers) in pbar:
            # Update progress bar description
            pbar.set_description(f"🔧 Testing Model {i+1}/{total_combinations}")
            pbar.set_postfix({
                'LR': lr, 
                'BS': batch_size, 
                'F': num_filters, 
                'L': num_layers
            })
            
            try:
                # Create and train model with GPU optimization
                model = self._create_and_train_model(
                    learning_rate=lr,
                    batch_size=batch_size,
                    num_filters=num_filters,
                    num_layers=num_layers,
                    epochs=config['tuning_epochs']
                )
                
                # Save individual model for this hyperparameter combination
                model_name = f"model_{i+1}_lr{lr}_bs{batch_size}_f{num_filters}_l{num_layers}"
                model_path = os.path.join(PATHS_CONFIG['models_dir'], f'{model_name}.keras')
                model.save(model_path)
                # Evaluate model
                evaluator = CNNEvaluator(model, self.class_names)
                metrics = evaluator.evaluate(self.test_data[0], self.test_data[1])
                
                # Store results with model name
                result = {
                    'model_name': f"Model {i+1}",
                    'model_file': f"{model_name}.keras",
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'num_filters': num_filters,
                    'num_layers': num_layers,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision_weighted'],
                    'recall': metrics['recall_weighted'],
                    'f1_score': metrics['f1_weighted'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.results.append(result)
                
                # Update best parameters
                if metrics['accuracy'] > self.best_score:
                    self.best_score = metrics['accuracy']
                    self.best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'num_filters': num_filters,
                        'num_layers': num_layers
                    }
                
                # Update progress bar with current best accuracy
                pbar.set_postfix({
                    'LR': lr, 
                    'BS': batch_size, 
                    'F': num_filters, 
                    'L': num_layers,
                    'Acc': f"{metrics['accuracy']:.3f}",
                    'Best': f"{self.best_score:.3f}"
                })
                
            except Exception as e:
                print(f"\n❌ Error with combination {i+1}: {e}")
                pbar.set_postfix({
                    'LR': lr, 
                    'BS': batch_size, 
                    'F': num_filters, 
                    'L': num_layers,
                    'Error': str(e)[:20]
                })
                continue
        
        # Close progress bar
        pbar.close()
        
        # Convert to DataFrame
        if not self.results:
            print("⚠️  No results collected! Creating empty DataFrame.")
            results_df = pd.DataFrame(columns=['model_name', 'model_file', 'learning_rate', 'batch_size', 
                                             'num_filters', 'num_layers', 'accuracy', 'precision', 
                                             'recall', 'f1_score', 'timestamp'])
        else:
            results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def _create_and_train_model(self, learning_rate, batch_size, num_filters, 
                               num_layers, epochs):
        """
        Create and train a model with given hyperparameters using GPU optimization
        
        Args:
            learning_rate: Learning rate
            batch_size: Batch size
            num_filters: Number of filters
            num_layers: Number of layers
            epochs: Number of epochs
        
        Returns:
            keras.Model: Trained model
        """
        # Create model
        cnn = CIFAR10CNN()
        model = cnn.build_model(num_filters=num_filters, num_layers=num_layers)
        model = cnn.compile_model(learning_rate=learning_rate)
        
        # GPU optimization: Get optimal batch size
        optimal_batch_size = get_optimal_batch_size(
            model, self.train_data[0], max_batch_size=batch_size * 4
        )
        
        # GPU optimization: Create optimized data pipeline
        train_dataset = create_optimized_data_pipeline(
            tf.data.Dataset.from_tensor_slices((self.train_data[0], self.train_data[1])),
            optimal_batch_size
        )
        
        # Create trainer
        trainer = CNNTrainer(model)
        
        # Train model with GPU-optimized data pipeline
        trainer.train(
            train_data=train_dataset,
            epochs=epochs,
            batch_size=optimal_batch_size,
            learning_rate=learning_rate,
            verbose=0  # Suppress training output for cleaner progress bar
        )
        
        return model
    
    def _save_results(self, results_df):
        """
        Save hyperparameter tuning results
        
        Args:
            results_df: Results DataFrame
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_path = os.path.join(PATHS_CONFIG['results_dir'], 
                               f'hyperparameter_results_{timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save summary
        summary_path = os.path.join(PATHS_CONFIG['results_dir'], 
                                   f'hyperparameter_summary_{timestamp}.txt')
        with open(summary_path, 'w') as f:
            f.write("Hyperparameter Tuning Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total combinations tested: {len(results_df)}\n")
            f.write(f"Best accuracy: {self.best_score:.4f}\n\n")
            
            if self.best_params:
                f.write("Best parameters:\n")
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            else:
                f.write("No best parameters found.\n")
            
            f.write("\nTop 10 Results:\n")
            f.write("-" * 30 + "\n")
            if len(results_df) > 0 and 'accuracy' in results_df.columns:
                top_results = results_df.nlargest(min(10, len(results_df)), 'accuracy')
                f.write(top_results.to_string(index=False))
            else:
                f.write("No results available.\n")
        
        print(f"Summary saved to {summary_path}")
    
    def plot_hyperparameter_analysis(self, results_df, save_plot=True):
        """
        Plot hyperparameter analysis
        
        Args:
            results_df: Results DataFrame
            save_plot: Whether to save plots
        
        Returns:
            list: List of matplotlib figures
        """
        figures = []
        
        # Check if we have results to plot
        if results_df.empty or 'accuracy' not in results_df.columns:
            print("⚠️  No results to plot. Skipping hyperparameter analysis plots.")
            return figures
        
        # 1. Accuracy vs Learning Rate
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        lr_accuracy = results_df.groupby('learning_rate')['accuracy'].agg(['mean', 'std'])
        ax1.errorbar(lr_accuracy.index, lr_accuracy['mean'], 
                    yerr=lr_accuracy['std'], marker='o', capsize=5)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Learning Rate')
        ax1.set_xscale('log')
        ax1.grid(True)
        figures.append(fig1)
        
        # 2. Accuracy vs Batch Size
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bs_accuracy = results_df.groupby('batch_size')['accuracy'].agg(['mean', 'std'])
        ax2.errorbar(bs_accuracy.index, bs_accuracy['mean'], 
                    yerr=bs_accuracy['std'], marker='o', capsize=5)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Batch Size')
        ax2.grid(True)
        figures.append(fig2)
        
        # 3. Accuracy vs Number of Filters
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        nf_accuracy = results_df.groupby('num_filters')['accuracy'].agg(['mean', 'std'])
        ax3.errorbar(nf_accuracy.index, nf_accuracy['mean'], 
                    yerr=nf_accuracy['std'], marker='o', capsize=5)
        ax3.set_xlabel('Number of Filters')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Number of Filters')
        ax3.grid(True)
        figures.append(fig3)
        
        # 4. Accuracy vs Number of Layers
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        nl_accuracy = results_df.groupby('num_layers')['accuracy'].agg(['mean', 'std'])
        ax4.errorbar(nl_accuracy.index, nl_accuracy['mean'], 
                    yerr=nl_accuracy['std'], marker='o', capsize=5)
        ax4.set_xlabel('Number of Layers')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Number of Layers')
        ax4.grid(True)
        figures.append(fig4)
        
        # 5. Heatmap of Learning Rate vs Batch Size
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        pivot_data = results_df.pivot_table(
            values='accuracy', 
            index='batch_size', 
            columns='learning_rate', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax5)
        ax5.set_title('Accuracy Heatmap: Learning Rate vs Batch Size')
        figures.append(fig5)
        
        # 6. Top 10 Results Bar Plot
        fig6, ax6 = plt.subplots(figsize=(12, 8))
        top_10 = results_df.nlargest(10, 'accuracy')
        x_labels = [f"LR:{row['learning_rate']}, BS:{row['batch_size']}, "
                   f"F:{row['num_filters']}, L:{row['num_layers']}" 
                   for _, row in top_10.iterrows()]
        ax6.bar(range(len(top_10)), top_10['accuracy'])
        ax6.set_xlabel('Hyperparameter Combinations')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Top 10 Hyperparameter Combinations')
        ax6.set_xticks(range(len(top_10)))
        ax6.set_xticklabels(x_labels, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        figures.append(fig6)
        
        if save_plot:
            for i, fig in enumerate(figures):
                plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 
                                       f'hyperparameter_analysis_{i+1}.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot {i+1} saved to {plot_path}")
        
        plt.show()
        return figures
    
    def create_comparison_table(self, results_df, save_table=True):
        """
        Create comprehensive comparison table showing all models with their settings
        
        Args:
            results_df: Results DataFrame
            save_table: Whether to save the table
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        # Sort by accuracy descending
        results_df_sorted = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
        
        # Create comprehensive comparison table
        comparison_data = []
        
        for i, (_, row) in enumerate(results_df_sorted.iterrows()):
            comparison_data.append({
                'Model': row['model_name'],
                'Model File': row['model_file'],
                'Learning Rate': row['learning_rate'],
                'Batch Size': row['batch_size'],
                'Filters': row['num_filters'],
                'Layers': row['num_layers'],
                'Accuracy': f"{row['accuracy']:.4f}",
                'Precision': f"{row['precision']:.4f}",
                'Recall': f"{row['recall']:.4f}",
                'F1-Score': f"{row['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*120)
        print("🏆 COMPREHENSIVE MODEL COMPARISON TABLE")
        print("="*120)
        print("📊 Hyperparameter Tuning Results (20 epochs each)")
        print("="*120)
        
        # Print model settings header
        print("\n📋 Model Settings:")
        print("-" * 80)
        for i, (_, row) in enumerate(results_df_sorted.iterrows()):
            print(f"{row['model_name']}: LR={row['learning_rate']}, BS={row['batch_size']}, "
                  f"Filters={row['num_filters']}, Layers={row['num_layers']}")
        
        print("\n📈 Performance Metrics:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        
        # Highlight best model
        best_model = results_df_sorted.iloc[0]
        print(f"\n🥇 BEST MODEL: {best_model['model_name']}")
        print(f"   Settings: LR={best_model['learning_rate']}, BS={best_model['batch_size']}, "
              f"Filters={best_model['num_filters']}, Layers={best_model['num_layers']}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        print(f"   Model File: {best_model['model_file']}")
        
        if save_table:
            table_path = os.path.join(PATHS_CONFIG['results_dir'], 
                                    'comprehensive_model_comparison.csv')
            comparison_df.to_csv(table_path, index=False)
            print(f"\n💾 Comprehensive comparison table saved to {table_path}")
        
        return comparison_df
    
    def get_best_model(self):
        """
        Get the best model based on hyperparameter tuning
        
        Returns:
            keras.Model: Best trained model
        """
        if not self.best_params:
            raise ValueError("No hyperparameter tuning results available")
        
        print("Training best model with optimal hyperparameters...")
        print(f"Best parameters: {self.best_params}")
        
        # Train model with best parameters
        best_model = self._create_and_train_model(
            learning_rate=self.best_params['learning_rate'],
            batch_size=self.best_params['batch_size'],
            num_filters=self.best_params['num_filters'],
            num_layers=self.best_params['num_layers'],
            epochs=50  # Use full epochs for final model
        )
        
        return best_model
