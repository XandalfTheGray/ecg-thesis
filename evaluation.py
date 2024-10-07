# evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
    coverage_error,
    confusion_matrix,
    classification_report
)
import os
import numpy as np
from matplotlib.colors import ListedColormap
from tensorflow import keras
import time

# Add the CustomProgressBar class here
class CustomProgressBar(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        self.seen = 0
        self.target = self.params['steps']

    def on_batch_end(self, batch, logs=None):
        self.seen += 1
        if self.seen % 10 == 0 or self.seen == self.target:
            print(f"\r{self.seen}/{self.target} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}", end='')

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\nval_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")
        print(f"Time: {epoch_time:.2f}s")

def create_output_directory(database, model):
    """
    Creates an output directory based on the database and model.
    
    Parameters:
    - database: str, the name of the database (e.g., 'mitbih')
    - model: str, the name of the model architecture (e.g., 'resnet18')
    
    Returns:
    - str, the path to the created output directory
    """
    output_dir = f'output_plots_{database}_{model}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def print_stats(predictions, labels):
    """
    Prints the performance statistics.

    Parameters:
    - predictions: array-like, predicted class labels.
    - labels: array-like, true class labels.
    """
    print("Accuracy = {0:.1f}%".format(accuracy_score(labels, predictions)*100))
    print("Precision = {0:.1f}%".format(precision_score(labels, predictions, average='macro')*100))
    print("Recall = {0:.1f}%".format(recall_score(labels, predictions, average='macro')*100))
    print("F1 Score = {0:.1f}%".format(f1_score(labels, predictions, average='macro')*100))

def showConfusionMatrix(predictions, labels, filename, output_dir, class_labels):
    """
    Plots and saves the confusion matrix with square cells, larger font, and no title.
    Labels are displayed right-side up for both axes.

    Parameters:
    - predictions: array-like, predicted class labels.
    - labels: array-like, true class labels.
    - filename: str, the name of the file to save the plot.
    - output_dir: str, the directory to save the plot.
    - class_labels: list, the list of class labels to display on the axes.
    """
    cfm_data = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased figure size for square shape
    cmap = ListedColormap(['white'])
    sns.heatmap(
        cfm_data, 
        annot=True, 
        fmt='d', 
        cmap=cmap, 
        cbar=False,
        linewidths=1, 
        linecolor='black', 
        xticklabels=class_labels,
        yticklabels=class_labels, 
        ax=ax,
        annot_kws={"size": 16}  # Increased font size for cell values
    )
    
    ax.set_ylabel('Actual Classification', fontsize=16)  # Increased font size
    ax.set_xlabel('Predicted Classification', fontsize=16)  # Increased font size
    
    # Remove title
    ax.set_title('')
    
    # Rotate x-axis labels to be right-side up and increase font size
    plt.xticks(rotation=0, fontsize=14)
    
    # Rotate y-axis labels to be right-side up and increase font size
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, output_dir):
    """
    Plots the training and validation loss and accuracy.

    Parameters:
    - history: History object from model.fit()
    - output_dir: str, the directory to save the plots
    """
    # Plot training & validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
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

def compute_and_save_multilabel_metrics(y_true, y_pred_classes, y_pred, label_names, output_dir):
    """
    Computes multilabel classification metrics, prints them, and saves to a text file.
    
    Parameters:
    - y_true (numpy.ndarray): True binary labels.
    - y_pred_classes (numpy.ndarray): Predicted binary labels.
    - y_pred (numpy.ndarray): Predicted probabilities.
    - label_names (list): List of label names.
    - output_dir (str): Directory to save the metrics file.
    """
    # Calculate Metrics
    h_loss = hamming_loss(y_true, y_pred_classes)
    exact_match = np.all(y_true == y_pred_classes, axis=1).mean()
    precision_micro = precision_score(y_true, y_pred_classes, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    
    recall_micro = recall_score(y_true, y_pred_classes, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    
    f1_micro = f1_score(y_true, y_pred_classes, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    
    jaccard = jaccard_score(y_true, y_pred_classes, average='macro', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        auc_roc = float('nan')  # Handle cases where AUC cannot be computed
    
    coverage = coverage_error(y_true, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred_classes, target_names=label_names)
    
    # Save Metrics to a File
    metrics_path = os.path.join(output_dir, 'additional_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Additional Metrics:\n")
        f.write(f"Hamming Loss: {h_loss:.4f}\n")
        f.write(f"Exact Match Ratio: {exact_match:.4f}\n")
        f.write(f"Precision (Micro): {precision_micro:.4f}\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}\n")
        f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (Micro): {recall_micro:.4f}\n")
        f.write(f"Recall (Macro): {recall_macro:.4f}\n")
        f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
        f.write(f"F1-Score (Micro): {f1_micro:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"Jaccard Index (Macro): {jaccard:.4f}\n")
        f.write(f"AUC-ROC (Macro): {auc_roc:.4f}\n")
        f.write(f"Coverage Error: {coverage:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
    
    print(f"Additional metrics and classification report saved to {metrics_path}")