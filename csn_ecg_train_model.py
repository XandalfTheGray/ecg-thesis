# File: csn_ecg_train_model.py
# This script trains a neural network model on the preprocessed CSN ECG dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.utils import class_weight
from tensorflow import keras
from datetime import datetime
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import importlib.util
import tensorflow as tf
from tensorflow.keras import mixed_precision
from google.colab import drive
import argparse

# Enable mixed precision for better performance on GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def setup_environment():
    """Set up the environment for Google Colab with mounted Google Drive."""
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive'
    print(f'Base Path: {base_path}')
    
    # Add the base path to sys.path to allow importing custom modules
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

def create_dataset(X, y, batch_size=32, shuffle=True, prefetch=True):
    """Create a TensorFlow dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main(time_steps):
    # Set up the environment and get the base path
    base_path = setup_environment()
    print(f"Base path set to: {base_path}")

    # Import custom modules
    sys.path.append(base_path)
    models = import_module('models')
    evaluation = import_module('evaluation')

    # Extract necessary functions from imported modules
    build_cnn = models.build_cnn
    build_resnet18_1d = models.build_resnet18_1d
    build_resnet34_1d = models.build_resnet34_1d
    build_resnet50_1d = models.build_resnet50_1d
    build_transformer = models.build_transformer
    print_stats = evaluation.print_stats
    showConfusionMatrix = evaluation.showConfusionMatrix

    # Set up output directories and model parameters
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csn_ecg'
    model_type = 'cnn'  # Options: 'cnn', 'resnet18', 'resnet34', 'resnet50', 'transformer'

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{time_steps}steps_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    learning_rate = 1e-3

    # Define model-specific parameters
    if model_type == 'transformer':
        model_params = {
            'head_size': 256,
            'num_heads': 4,
            'ff_dim': 4,
            'num_transformer_blocks': 4,
            'mlp_units': [128],
            'mlp_dropout': 0.4,
            'dropout': 0.25,
        }
    else:
        model_params = {
            'l2_reg': 0.001,
            'filters': [32, 64, 128],
            'kernel_sizes': [5, 5, 5],
            'dropout_rates': [0.3, 0.3, 0.3, 0.3],
        }

    # Load preprocessed data for the specified time steps
    try:
        data_dir = os.path.join(base_path, 'csnecg_preprocessed_data', f'{time_steps}_signal_time_steps')
        X = np.load(os.path.join(data_dir, 'X.npy'))
        Y_cl = np.load(os.path.join(data_dir, 'Y.npy'), allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Ensure that the preprocessed data for {time_steps} time steps exists in the 'csnecg_preprocessed_data' folder in your Google Drive.")
        sys.exit(1)

    # Print data summary
    print(f"Loaded data shape - X: {X.shape}, Y_cl: {len(Y_cl)}")
    unique_classes = set(class_name for sublist in Y_cl for class_name in sublist)
    print(f"Unique classes: {unique_classes}")

    # Analyze initial class distribution
    initial_class_counts = Counter([item for sublist in Y_cl for item in sublist])
    print("Initial class distribution:", dict(initial_class_counts))

    # Binarize the labels for multi-label classification
    mlb = MultiLabelBinarizer(classes=sorted(unique_classes))
    Y_binarized = mlb.fit_transform(Y_cl)

    # Create label mappings
    label_names = mlb.classes_
    Num2Label = {idx: label for idx, label in enumerate(label_names)}
    label2Num = {label: idx for idx, label in enumerate(label_names)}
    num_classes = len(label_names)

    print("\nLabel to Number Mapping:")
    for label, num in label2Num.items():
        print(f"{label}: {num}")

    print(f"\nNumber of Classes: {num_classes}")

    # Print updated class distribution
    updated_class_counts = Counter([item for sublist in Y_cl for item in sublist])
    print("Updated class distribution:", dict(updated_class_counts))

    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y_binarized: {Y_binarized.shape}")

    # Check for samples with insufficient data
    samples_per_class = Y_binarized.sum(axis=0)
    min_samples_per_class = samples_per_class.min()
    print(f"Samples per class: {samples_per_class}")
    print(f"Minimum samples for any class: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("Error: There are still classes with less than 2 samples.")
        return

    # Remove samples with no positive labels
    samples_with_no_labels = (Y_binarized.sum(axis=1) == 0).sum()
    if samples_with_no_labels > 0:
        print(f"Warning: {samples_with_no_labels} samples have no positive labels.")
        valid_samples = Y_binarized.sum(axis=1) > 0
        X = X[valid_samples]
        Y_binarized = Y_binarized[valid_samples]
        print(f"Shape after removing samples with no labels - X: {X.shape}, Y_binarized: {Y_binarized.shape}")

    # Analyze and remove rare label combinations
    label_combinations = [tuple(row) for row in Y_binarized]
    combination_counts = Counter(label_combinations)
    print("Label combination counts:", dict(combination_counts))

    min_combination_count = 2
    valid_combinations = [comb for comb, count in combination_counts.items() if count >= min_combination_count]
    valid_indices = [i for i, comb in enumerate(label_combinations) if comb in valid_combinations]

    X = X[valid_indices]
    Y_binarized = Y_binarized[valid_indices]

    print(f"Shape after removing rare combinations - X: {X.shape}, Y_binarized: {Y_binarized.shape}")

    # Split the data into train, validation, and test sets
    test_size = 0.2
    val_size = 0.25  # 25% of the remaining 80% = 20% of total

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y_binarized, test_size=test_size, random_state=42, stratify=Y_binarized
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
    except ValueError as e:
        print(f"Error during train_test_split: {str(e)}")
        print("Attempting split without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y_binarized, test_size=test_size, random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

    # Scale the data
    def scale_dataset(X, scaler):
        num_samples, num_timesteps, num_channels = X.shape
        X_reshaped = X.reshape(-1, num_channels)
        X_scaled = scaler.transform(X_reshaped)
        return X_scaled.reshape(num_samples, num_timesteps, num_channels)

    num_samples_train, num_timesteps, num_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_channels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(num_samples_train, num_timesteps, num_channels)

    X_valid_scaled = scale_dataset(X_valid, scaler)
    X_test_scaled = scale_dataset(X_test, scaler)

    print(f"\nTrain set shape: {X_train_scaled.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_valid_scaled.shape}, {y_valid.shape}")
    print(f"Test set shape: {X_test_scaled.shape}, {y_test.shape}")

    batch_size = 128
    train_dataset = create_dataset(X_train_scaled, y_train, batch_size=batch_size)
    valid_dataset = create_dataset(X_valid_scaled, y_valid, batch_size=batch_size, shuffle=False)
    test_dataset = create_dataset(X_test_scaled, y_test, batch_size=batch_size, shuffle=False)

    class_weights = {}
    for i, class_label in enumerate(label_names):
        class_count = y_train[:, i].sum()
        if class_count == 0:
            class_weights[i] = 1.0
        else:
            class_weights[i] = (len(y_train) / (num_classes * class_count))

    if model_type == 'cnn':
        model = build_cnn(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif model_type == 'resnet18':
        model = build_resnet18_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif model_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif model_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif model_type == 'transformer':
        model = build_transformer(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    else:
        raise ValueError("Invalid model type.")

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=valid_dataset,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print("\nEvaluating the model on the test set:")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(test_dataset)
    y_pred_classes = (y_pred > 0.5).astype(int)

    assert len(y_pred_classes) == len(y_test), "Mismatch between predictions and test labels"

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_names))

    plot_confusion_matrices(y_test, y_pred_classes, label_names, output_dir)
    plot_training_history(history, output_dir)

    print(f"\nTraining completed. Results saved in {output_dir}")

    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Time Steps: {time_steps}\n")
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
    parser = argparse.ArgumentParser(description='Train a neural network model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, required=True, help='Number of time steps in the preprocessed data.')
    args = parser.parse_args()
    main(args.time_steps)