# evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    roc_auc_score,
    coverage_error,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve
)
import logging
import time
from datetime import datetime
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# Multiclass Evaluation and Plotting Functions
# ==============================

def print_stats(y_pred, y_true):
    """
    Prints custom statistics based on predictions and true labels.
    
    Parameters:
    - y_pred (np.ndarray): Predicted class labels.
    - y_true (np.ndarray): True class labels.
    """
    # Example implementation; replace with your actual statistics
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy:.4f}")

def showConfusionMatrix(y_pred, y_true, filename, output_dir, label_names):
    """
    Plots and saves a confusion matrix.
    
    Parameters:
    - y_pred (np.ndarray): Predicted class labels.
    - y_true (np.ndarray): True class labels.
    - filename (str): Filename for saving the confusion matrix plot.
    - output_dir (str): Directory to save the plot.
    - label_names (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.heatmap(cm, annot=True, fmt='d', cmap=ListedColormap(['white']),
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor='black', cbar=False)
    # Add outer border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

class CustomProgressBar(Callback):
    """
    Custom callback to display progress bars during training.
    """
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        self.seen = 0
        
        # Get or calculate total steps
        self.total_steps = self.params.get('steps', None)
        if self.total_steps is None:
            if 'samples' in self.params and 'batch_size' in self.params and self.params['batch_size'] is not None:
                self.total_steps = int(np.ceil(self.params['samples'] / self.params['batch_size']))
            else:
                self.total_steps = 0

        self.progbar = tf.keras.utils.Progbar(target=self.total_steps)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1
        
        if self.total_steps > 0:
            # Get all available metrics from logs
            values = [(k, v) for k, v in logs.items() if not k.startswith('val_')]
            self.progbar.update(self.seen, values=values)

# ==============================
# Multilabel Evaluation and Plotting Functions
# ==============================

def compute_metrics(y_true, y_pred, y_scores):
    """
    Computes multilabel classification metrics.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - y_scores (np.ndarray): Predicted scores/probabilities.

    Returns:
    - metrics_dict (dict): Dictionary containing all computed metrics.
    """
    metrics_dict = {}
    
    # Basic Metrics
    metrics_dict['Hamming Loss'] = hamming_loss(y_true, y_pred)
    metrics_dict['Exact Match Ratio'] = np.all(y_true == y_pred, axis=1).mean()
    
    # Precision, Recall, F1-Score
    metrics_dict['Precision (Micro)'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics_dict['Precision (Macro)'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['Precision (Weighted)'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics_dict['Recall (Micro)'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics_dict['Recall (Macro)'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['Recall (Weighted)'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics_dict['F1-Score (Micro)'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics_dict['F1-Score (Macro)'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['F1-Score (Weighted)'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Jaccard Index
    metrics_dict['Jaccard Index (Macro)'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC-ROC
    try:
        metrics_dict['AUC-ROC (Macro)'] = roc_auc_score(y_true, y_scores, average='macro')
    except ValueError:
        metrics_dict['AUC-ROC (Macro)'] = np.nan  # Handle cases where AUC cannot be computed
    
    # Coverage Error
    metrics_dict['Coverage Error'] = coverage_error(y_true, y_scores)
    
    return metrics_dict

def plot_confusion_matrices_grid(y_true, y_pred, label_names, output_dir):
    """
    Plots and saves confusion matrices for each class and the overall confusion matrix in a grid layout.
    
    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - label_names (list): List of class names.
    - output_dir (str): Directory to save the plot.
    """
    num_classes = len(label_names)
    fig_rows, fig_cols = 2, 3  # Adjust grid size as needed
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(18, 12))
    axes = axes.ravel()

    # Plot confusion matrices for each class
    for idx, label in enumerate(label_names):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap=ListedColormap(['white']),
                    xticklabels=[f'Not {label}', label],
                    yticklabels=[f'Not {label}', label],
                    linewidths=0.5, linecolor='black', ax=ax, cbar=False)
        ax.set_title(f'Confusion Matrix for {label}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        # Add outer border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('black')

    # Plot overall confusion matrix in the last subplot
    cm_total = np.zeros((2, 2), dtype=int)
    for idx in range(num_classes):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        cm_total += cm

    ax = axes[-1]
    sns.heatmap(cm_total, annot=True, fmt='d', cmap=ListedColormap(['white']),
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                linewidths=0.5, linecolor='black', ax=ax)
    ax.set_title('Overall Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    # Add outer border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')

    # Hide any unused subplots
    total_plots = fig_rows * fig_cols
    if num_classes < total_plots - 1:
        for i in range(num_classes, total_plots - 1):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_grid.png'))
    plt.close()

def plot_precision_recall_per_class(y_true, y_scores, label_names, output_dir):
    """
    Plots and saves Precision-Recall curves for each class.
    """
    for idx, label in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true[:, idx], y_scores[:, idx])
        average_precision = average_precision_score(y_true[:, idx], y_scores[:, idx])
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'AP={average_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {label}')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'precision_recall_{label}.png'))
        plt.close()

def plot_precision_recall_aggregated(y_true, y_scores, label_names, output_dir):
    """
    Plots and saves aggregated Precision-Recall curves for all classes on a single graph.
    """
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true[:, idx], y_scores[:, idx])
        average_precision = average_precision_score(y_true[:, idx], y_scores[:, idx])
        plt.plot(recall, precision, label=f'{label} (AP={average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Aggregated Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_precision_recall.png'))
    plt.close()

def plot_roc_curve_per_class(y_true, y_scores, label_names, output_dir):
    """
    Plots and saves ROC curves for each class.
    """
    for idx, label in enumerate(label_names):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, idx], y_scores[:, idx])
            roc_auc = roc_auc_score(y_true[:, idx], y_scores[:, idx])
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {label}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'roc_curve_{label}.png'))
            plt.close()
        except ValueError:
            logging.warning(f"ROC AUC score cannot be computed for class {label}. Skipping.")

def plot_roc_curve_aggregated(y_true, y_scores, label_names, output_dir):
    """
    Plots and saves aggregated ROC curves for all classes on a single graph.
    """
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(label_names):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, idx], y_scores[:, idx])
            roc_auc = roc_auc_score(y_true[:, idx], y_scores[:, idx])
            plt.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.2f})')
        except ValueError:
            logging.warning(f"ROC AUC score cannot be computed for class {label}. Skipping.")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Aggregated ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_roc_curves.png'))
    plt.close()

def plot_metrics_bar_chart(y_true, y_pred, label_names, output_dir):
    """
    Plots and saves bar charts for F1-Score and Jaccard Index for each class.
    """
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(label_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, f1, width, label='F1-Score')
    plt.bar(x + width/2, jaccard, width, label='Jaccard Index')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('F1-Score and Jaccard Index per Class')
    plt.xticks(x, label_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_jaccard_bar_chart.png'))
    plt.close()

def plot_label_distribution(y_true, label_names, output_dir):
    """
    Plots and saves a histogram of label distribution.
    """
    label_counts = y_true.sum(axis=0)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_names, y=label_counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()

def plot_training_history(history, output_dir):
    """Plot and save the training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def evaluate_multilabel_model(y_true, y_pred, y_scores, label_names, output_dir, history=None):
    """
    Computes metrics, generates classification report, and creates evaluation plots.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - y_scores (np.ndarray): Predicted scores/probabilities.
    - label_names (list): List of class names.
    - output_dir (str): Directory to save evaluation results.
    - history (keras.callbacks.History, optional): History object from model training.

    Returns:
    - metrics_dict (dict): Dictionary containing all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    metrics_dict = compute_metrics(y_true, y_pred, y_scores)
    
    # Print classification report
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save additional metrics
    with open(os.path.join(output_dir, 'additional_metrics.txt'), 'w') as f:
        f.write("Additional Metrics:\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Generate individual plots
    # plot_confusion_matrix_per_class(y_true, y_pred, label_names, output_dir)
    plot_confusion_matrices_grid(y_true, y_pred, label_names, output_dir)
    plot_precision_recall_per_class(y_true, y_scores, label_names, output_dir)
    plot_roc_curve_per_class(y_true, y_scores, label_names, output_dir)
    
    # Generate aggregated plots
    plot_precision_recall_aggregated(y_true, y_scores, label_names, output_dir)
    plot_roc_curve_aggregated(y_true, y_scores, label_names, output_dir)
    
    # Generate additional visualizations
    plot_metrics_bar_chart(y_true, y_pred, label_names, output_dir)
    plot_label_distribution(y_true, label_names, output_dir)
    
    # Plot training history if history is provided
    if history is not None:
        plot_training_history(history, output_dir)
    
    logging.info(f"Evaluation metrics and plots saved in '{output_dir}'")
    
    return metrics_dict

class TimingCallback(keras.callbacks.Callback):
    """Callback for tracking training time per epoch."""
    def on_train_begin(self, logs=None):
        self.times = []
        self.start_time = None
        self.total_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.times.append(elapsed_time)
            if logs is not None:
                logs['time'] = elapsed_time

    def get_total_time(self):
        return time.time() - self.total_start_time

def log_timing_info(timing_callback, model_info, output_dir):
    """
    Log timing information to a file.
    
    Args:
        timing_callback: TimingCallback instance with timing data
        model_info: Dict containing model information (type, parameters, etc.)
        output_dir: Directory to save the timing log
    """
    log_file = os.path.join(output_dir, 'timing_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Model Type: {model_info.get('model_type', 'Not specified')}\n")
        f.write(f"Dataset: {model_info.get('dataset', 'Not specified')}\n\n")
        
        f.write("Training Timing Information:\n")
        f.write(f"Total Training Time: {timing_callback.get_total_time():.2f} seconds\n")
        f.write(f"Average Time per Epoch: {np.mean(timing_callback.times):.2f} seconds\n")
        f.write(f"Fastest Epoch: {np.min(timing_callback.times):.2f} seconds\n")
        f.write(f"Slowest Epoch: {np.max(timing_callback.times):.2f} seconds\n")
        f.write(f"Number of Epochs: {len(timing_callback.times)}\n\n")
        
        f.write("Per-Epoch Timing:\n")
        for epoch, time_taken in enumerate(timing_callback.times, 1):
            f.write(f"Epoch {epoch}: {time_taken:.2f} seconds\n")

def print_detailed_stats(y_pred, y_true):
    """Print detailed classification metrics."""
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
