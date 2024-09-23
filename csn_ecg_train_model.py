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

# Import models and evaluation functions
from models import build_cnn, build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import print_stats, showConfusionMatrix

# Import data preprocessing functions
from csn_ecg_data_preprocessing import load_data as load_csn_data, load_snomed_ct_mapping

def main():
    # Setup parameters
    base_output_dir = 'output_plots'
    dataset_name = 'csn_ecg'
    model_type = 'cnn'  # Options: 'cnn', 'resnet18', 'resnet34', 'resnet50'

    # Create a unique output directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
        'filters': (32, 64, 128),
        'kernel_sizes': (3, 3, 3),
        'dropout_rates': (0.1, 0.1, 0.1, 0.5)
    }
    
    # Set up paths for the CSN ECG dataset
    database_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 
                                 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0')
    wfdb_dir = os.path.join(database_path, 'WFDBRecords')
    
    if not os.path.exists(wfdb_dir):
        print(f"Error: Directory {wfdb_dir} does not exist.")
        return

    # Gather all record names
    data_entries = []
    for subdir, dirs, files in os.walk(wfdb_dir):
        for file in files:
            if file.endswith('.mat'):
                record_path = os.path.join(subdir, file)
                record_name = os.path.relpath(record_path, wfdb_dir)
                record_name = os.path.splitext(record_name)[0]  # Remove the .mat extension
                data_entries.append(record_name)

    print(f"Total records found for CSN ECG: {len(data_entries)}")
    
    if len(data_entries) == 0:
        print("Error: No records found. Check the database path and file structure.")
        return

    # Load and preprocess data
    max_records = 5000
    print(f"Processing up to {max_records} records for CSN ECG dataset")
    csv_path = os.path.join(database_path, 'ConditionNames_SNOMED-CT.csv')

    # Specify the conditions you want to classify
    selected_conditions = [
        'Sinus Rhythm',
        'Atrial Fibrillation',
        'Atrial Flutter',
        'Sinus Bradycardia',
        'Sinus Tachycardia'
    ]

    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, selected_conditions)
    X, Y_cl = load_csn_data(wfdb_dir, data_entries, snomed_ct_mapping, max_records=max_records, desired_length=5000)

    if X.size == 0 or len(Y_cl) == 0:
        print("Error: No data was loaded. Check the data preprocessing step.")
        return

    # Print data summary
    print(f"Loaded data shape - X: {X.shape}, Y_cl: {len(Y_cl)}")
    unique_codes = set(code for sublist in Y_cl for code in sublist)
    print(f"Unique SNOMED-CT codes: {unique_codes}")

    # Analyze class distribution
    # Flatten Y_cl for counting
    flat_Y = [code for sublist in Y_cl for code in sublist]
    class_counts = Counter(flat_Y)
    print("Class distribution:", dict(class_counts))

    # Filter out classes with less than 10 samples
    min_samples_per_class = 10
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples_per_class}

    # Include 'Unknown' only if it has sufficient samples
    if 'Unknown' in class_counts and class_counts['Unknown'] < min_samples_per_class:
        valid_classes.discard('Unknown')

    # Binarize the labels for multi-label classification
    mlb = MultiLabelBinarizer(classes=sorted(valid_classes))
    Y_binarized = mlb.fit_transform(Y_cl)

    # Update snomed_ct_mapping to include only valid classes
    label_names = mlb.classes_
    Num2Label = {idx: label for idx, label in enumerate(label_names)}
    label2Num = {label: idx for idx, label in enumerate(label_names)}
    num_classes = len(label_names)

    print("\nLabel to Number Mapping (first 10):")
    for label, num in list(label2Num.items())[:10]:
        print(f"{label}: {num}")

    print("\nNumber to Label Mapping (first 10):")
    for num, label in list(Num2Label.items())[:10]:
        print(f"{num}: {label}")

    print(f"\nNumber of Classes: {num_classes}")

    # Split the data into train, validation, and test sets (before scaling)
    test_size = 0.2
    val_size = 0.25  # 25% of the remaining 80% = 20% of total

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_binarized, test_size=test_size, random_state=42, stratify=Y_binarized
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )

    # Reshape and Scale data
    def scale_dataset(X, scaler):
        num_samples, num_timesteps, num_channels = X.shape
        X_reshaped = X.reshape(-1, num_channels)
        X_scaled = scaler.transform(X_reshaped)
        return X_scaled.reshape(num_samples, num_timesteps, num_channels)

    # Fit the scaler only on training data
    num_samples_train, num_timesteps, num_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_channels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(num_samples_train, num_timesteps, num_channels)

    # Apply the scaler to validation and test sets
    X_valid_scaled = scale_dataset(X_valid, scaler)
    X_test_scaled = scale_dataset(X_test, scaler)

    # Print the final shapes
    print(f"\nTrain set shape: {X_train_scaled.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_valid_scaled.shape}, {y_valid.shape}")
    print(f"Test set shape: {X_test_scaled.shape}, {y_test.shape}")

    # Compute class weights for imbalanced dataset
    # For multi-label, compute class weights individually
    class_weights = {}
    for i, class_label in enumerate(label_names):
        # Compute class weight based on the frequency of each class
        class_count = y_train[:, i].sum()
        if class_count == 0:
            class_weights[i] = 1.0
        else:
            class_weights[i] = (len(y_train) / (num_classes * class_count))
    # Alternatively, use sklearn's compute_class_weight for each class
    # This implementation is a simplified approach

    # Build the neural network model
    if model_type == 'cnn':
        model = build_cnn(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            **model_params
        )
    elif model_type == 'resnet18':
        model = build_resnet18_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            l2_reg=model_params['l2_reg']
        )
    else:
        raise ValueError("Invalid model type.")

    # Compile the model
    model.compile(
        loss='binary_crossentropy',  # For multi-label
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    # Define callbacks for training
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    # Train the model using scaled data
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=30, 
        validation_data=(X_valid_scaled, y_valid),
        batch_size=256, 
        shuffle=True, 
        callbacks=callbacks, 
        class_weight=class_weights
    )

    # Define evaluation function
    def evaluate_model(dataset, y_true, name):
        y_pred_prob = model.predict(dataset)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        print(f"\n{name} Performance")
        print_stats(y_pred, y_true)
        showConfusionMatrix(
            y_pred, y_true, f'confusion_matrix_{name.lower()}.png', output_dir, list(Num2Label.values())
        )

    # Evaluate the model on scaled data
    evaluate_model(X_train_scaled, y_train, 'Training')
    evaluate_model(X_valid_scaled, y_valid, 'Validation')
    evaluate_model(X_test_scaled, y_test, 'Test')

    # Plot training history
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

if __name__ == '__main__':
    main()
