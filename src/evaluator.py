"""
Model evaluation module for CNN CIFAR-10

FILE PURPOSE:
    Comprehensive model evaluation and analysis for CNN CIFAR-10 classification.
    Provides detailed performance metrics, confusion matrix analysis, per-class
    evaluation, misclassification analysis, and generates comprehensive reports.

DEPENDENCIES (IMPORTS FROM):
    - numpy: Numerical computing for data manipulation
    - matplotlib.pyplot: Visualization plotting
    - seaborn: Statistical data visualization
    - sklearn.metrics: Classification metrics (confusion_matrix, classification_report, etc.)
    - pandas: Data manipulation and analysis
    - os: File system operations
    - configs.config: PATHS_CONFIG, DATASET_CONFIG for paths and class names

OUTPUTS (GENERATES):
    - Performance metrics: Accuracy, precision, recall, F1-score
    - plots/confusion_matrix.png: Confusion matrix visualization
    - plots/class_metrics.png: Per-class performance metrics
    - plots/misclassifications.png: Error analysis visualization
    - results/classification_report.txt: Detailed text report
    - results/performance_metrics.csv: Quantitative metrics table

ROLE IN PROJECT:
    Model assessment component that evaluates trained CNN performance.
    Used by main.py to analyze model quality and generate evaluation reports.
    Provides comprehensive analysis tools for model validation and comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_recall_fscore_support)
import pandas as pd
import os
from configs.config import PATHS_CONFIG, DATASET_CONFIG

class CNNEvaluator:
    """
    Evaluator class for CNN model performance analysis
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or DATASET_CONFIG['class_names']
        self.predictions = None
        self.true_labels = None
        
    def evaluate(self, test_images, test_labels, batch_size=32):
        """
        Evaluate model on test data
        
        Args:
            test_images: Test images
            test_labels: True test labels
            batch_size: Batch size for evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Get predictions
        self.predictions = self.model.predict(test_images, batch_size=batch_size)
        self.true_labels = test_labels
        
        # Convert predictions to class labels
        predicted_labels = np.argmax(self.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predicted_labels, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            test_labels, predicted_labels, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'predicted_labels': predicted_labels
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision (weighted): {precision:.4f}")
        print(f"Test Recall (weighted): {recall:.4f}")
        print(f"Test F1-Score (weighted): {f1:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_plot=True, figsize=(10, 8)):
        """
        Plot confusion matrix
        
        Args:
            save_plot: Whether to save the plot
            figsize: Figure size
        
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must evaluate model first before plotting confusion matrix")
        
        predicted_labels = np.argmax(self.predictions, axis=1)
        cm = confusion_matrix(self.true_labels, predicted_labels)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - CNN CIFAR-10')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {plot_path}")
        
        plt.show()
        return plt.gcf()
    
    def plot_class_metrics(self, metrics, save_plot=True, figsize=(15, 5)):
        """
        Plot per-class metrics
        
        Args:
            metrics: Evaluation metrics dictionary
            save_plot: Whether to save the plot
            figsize: Figure size
        
        Returns:
            matplotlib.figure.Figure: Class metrics plot
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Precision
        axes[0].bar(self.class_names, metrics['precision_per_class'])
        axes[0].set_title('Precision per Class')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1].bar(self.class_names, metrics['recall_per_class'])
        axes[1].set_title('Recall per Class')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[2].bar(self.class_names, metrics['f1_per_class'])
        axes[2].set_title('F1-Score per Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 'class_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Class metrics plot saved to {plot_path}")
        
        plt.show()
        return fig
    
    def generate_classification_report(self, save_report=True):
        """
        Generate detailed classification report
        
        Args:
            save_report: Whether to save the report
        
        Returns:
            str: Classification report
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must evaluate model first before generating report")
        
        predicted_labels = np.argmax(self.predictions, axis=1)
        
        report = classification_report(
            self.true_labels, predicted_labels,
            target_names=self.class_names,
            digits=4
        )
        
        print("Classification Report:")
        print("=" * 50)
        print(report)
        
        if save_report:
            report_path = os.path.join(PATHS_CONFIG['results_dir'], 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write("CNN CIFAR-10 Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            print(f"Classification report saved to {report_path}")
        
        return report
    
    def create_metrics_table(self, metrics, model_name="CNN", save_table=True):
        """
        Create performance metrics table
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            save_table: Whether to save the table
        
        Returns:
            pandas.DataFrame: Metrics table
        """
        # Create table data
        table_data = {
            'Model': [model_name],
            'Accuracy': [f"{metrics['accuracy']:.4f}"],
            'Precision': [f"{metrics['precision_weighted']:.4f}"],
            'Recall': [f"{metrics['recall_weighted']:.4f}"],
            'F1-Score': [f"{metrics['f1_weighted']:.4f}"]
        }
        
        df = pd.DataFrame(table_data)
        
        print("\nPerformance Metrics Table:")
        print("=" * 50)
        print(df.to_string(index=False))
        
        if save_table:
            table_path = os.path.join(PATHS_CONFIG['results_dir'], 'performance_metrics.csv')
            df.to_csv(table_path, index=False)
            print(f"Performance metrics table saved to {table_path}")
        
        return df
    
    def analyze_misclassifications(self, test_images, num_samples=10, save_plot=True):
        """
        Analyze and visualize misclassified samples
        
        Args:
            test_images: Test images
            num_samples: Number of misclassified samples to show
            save_plot: Whether to save the plot
        
        Returns:
            matplotlib.figure.Figure: Misclassification analysis plot
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Must evaluate model first before analyzing misclassifications")
        
        predicted_labels = np.argmax(self.predictions, axis=1)
        
        # Find misclassified samples
        misclassified_indices = np.where(predicted_labels != self.true_labels)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return None
        
        # Select random misclassified samples
        np.random.seed(42)
        selected_indices = np.random.choice(
            misclassified_indices, 
            min(num_samples, len(misclassified_indices)), 
            replace=False
        )
        
        # Create subplot
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_indices):
            if i >= 10:
                break
                
            # Denormalize image for display
            img = test_images[idx]
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            
            axes[i].imshow(img)
            axes[i].set_title(f'True: {self.class_names[self.true_labels[idx]]}\n'
                            f'Pred: {self.class_names[predicted_labels[idx]]}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(selected_indices), 10):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(PATHS_CONFIG['plots_dir'], 'misclassifications.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Misclassification analysis saved to {plot_path}")
        
        plt.show()
        return fig
