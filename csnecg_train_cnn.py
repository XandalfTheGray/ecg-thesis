# File: csnecg_train_cnn.py
# This script trains a CNN model on the preprocessed CSN ECG dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import importlib.util
import tensorflow as tf
from google.colab import drive
import argparse

def setup_environment():
    """Set up the environment for Google Colab with mounted Google Drive."""
    base_path = '/content/drive/MyDrive'
    print(f'Base Path: {base_path}')
    
    if not os.path.exists(base_path):
        raise RuntimeError("Google Drive is not mounted. Please mount it manually before running this script.")
    
    if base_path not in sys.path:
        sys.path.append(base_path)
    
    return base_path

def import_module(module_name):
    """Dynamically import a module by name."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing {module_name}. Make sure the file is in the correct directory.")
        sys.exit(1)

def main(time_steps, batch_size):
    # Set up the environment and import necessary modules
    base_path = setup_environment()
    print(f"Base path set to: {base_path}")

    sys.path.append(base_path)
    models = import_module('models')
    evaluation = import_module('evaluation')
    csn_ecg_data_preprocessing = import_module('csn_ecg_data_preprocessing')

    # Import specific functions from modules
    build_cnn = models.build_cnn
    print_stats = evaluation.print_stats
    showConfusionMatrix = evaluation.showConfusionMatrix
    prepare_csn_ecg_data = csn_ecg_data_preprocessing.prepare_csn_ecg_data

    # Set up output directories
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csn_ecg'
    model_type = 'cnn'
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{time_steps}steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)
    
    learning_rate = 1e-3

    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
        'filters': [32, 64, 128],
        'kernel_sizes': [5, 5, 5],
        'dropout_rates': [0.3, 0.3, 0.3, 0.3],
    }

    # Prepare data
    train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csn_ecg_data(time_steps, base_path, batch_size)

    # Build the CNN model
    model = build_cnn(
        input_shape=(time_steps, 12),
        num_classes=num_classes,
        activation='sigmoid',
        **model_params
    )

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )

    # Define callbacks for training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    # Evaluate the model on the test set
    print("\nEvaluating the model on the test set:")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate predictions and classification report
    y_pred = model.predict(test_dataset)
    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=label_names))

    # Plot confusion matrices and training history
    plot_confusion_matrices(y_true, y_pred_classes, label_names, output_dir)
    plot_training_history(history, output_dir)

    print(f"\nTraining completed. Results saved in {output_dir}")

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Time Steps: {time_steps}\n")
        f.write(f"Batch Size: {batch_size}\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrices for each class."""
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {class_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{class_name}.png'))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, choices=[500, 1000, 2000, 5000], required=True, 
                        help='Number of time steps in the preprocessed data.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    args = parser.parse_args()
    main(args.time_steps, args.batch_size)