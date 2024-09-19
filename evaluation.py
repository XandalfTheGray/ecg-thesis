# evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score,
                             confusion_matrix, 
                             f1_score)
import os

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
    Plots and saves the confusion matrix.

    Parameters:
    - predictions: array-like, predicted class labels.
    - labels: array-like, true class labels.
    - filename: str, the name of the file to save the plot.
    - output_dir: str, the directory to save the plot.
    - class_labels: list, the list of class labels to display on the axes.
    """
    # Create Confusion Matrix
    cfm_data = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    cf_matrix = sns.heatmap(cfm_data, annot=True, fmt='.0f', square=True,
                            cmap='YlGnBu', linewidths=0.5, linecolor='k', cbar=False)

    # Apply Axis Formatting
    cf_matrix.set_xlabel("Predicted Classification")
    cf_matrix.set_ylabel("Actual Classification")
    cf_matrix.xaxis.set_ticklabels(class_labels)
    cf_matrix.yaxis.set_ticklabels(class_labels)

    # Save Confusion Matrix to file
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
