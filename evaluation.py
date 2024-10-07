# evaluation.py
# This script contains functions to evaluate and visualize the performance of multilabel classification models.

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

class CustomProgressBar(keras.callbacks.Callback):
    """
    Custom callback to display progress bars during training.
    """
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs.get('loss'):.4f}, Accuracy = {logs.get('accuracy'):.4f}")

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

def plot_confusion_matrix_per_class(y_true, y_pred, label_names, output_dir):
    """
    Plots and saves a confusion matrix for each class in a multilabel setting.
    """
    for idx, label in enumerate(label_names):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {label}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{label}.png'))
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

def plot_roc_curve_per_class(y_true, y_scores, label_names, output_dir):
    """
    Plots and saves ROC curves for each class.
    """
    for idx, label in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_scores[:, idx])
        try:
            roc_auc = roc_auc_score(y_true[:, idx], y_scores[:, idx])
        except ValueError:
            roc_auc = np.nan  # Handle cases where AUC cannot be computed
        
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

def plot_confusion_matrix_overall(y_true, y_pred, label_names, output_dir):
    """
    Plots and saves an overall confusion matrix aggregated across all classes.
    """
    # Aggregate confusion matrices for all classes
    cm_total = np.zeros((2,2))
    for idx in range(y_true.shape[1]):
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        cm_total += cm
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_confusion_matrix.png'))
    plt.close()

def evaluate_multilabel_model(y_true, y_pred, y_scores, label_names, output_dir):
    """
    Computes metrics, generates classification report, and creates evaluation plots.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted binary labels.
    - y_scores (np.ndarray): Predicted scores/probabilities.
    - label_names (list): List of class names.
    - output_dir (str): Directory to save evaluation results.

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
    
    # Generate plots
    plot_confusion_matrix_per_class(y_true, y_pred, label_names, output_dir)
    plot_precision_recall_per_class(y_true, y_scores, label_names, output_dir)
    plot_roc_curve_per_class(y_true, y_scores, label_names, output_dir)
    plot_metrics_bar_chart(y_true, y_pred, label_names, output_dir)
    plot_label_distribution(y_true, label_names, output_dir)
    plot_confusion_matrix_overall(y_true, y_pred, label_names, output_dir)
    
    logging.info(f"Evaluation metrics and plots saved in '{output_dir}'")
    
    return metrics_dict