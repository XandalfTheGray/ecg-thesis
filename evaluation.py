# evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score,
                             confusion_matrix, 
                             f1_score)
import os
import numpy as np
from matplotlib.colors import ListedColormap

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

# Example usage:
# output_dir = create_output_directory('mitbih', 'resnet18')
# showConfusionMatrix(predictions, labels, 'confusion_matrix.png', output_dir, class_labels)
# plot_training_history(history, output_dir)
