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
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import seaborn as sns
import sys
import importlib.util
from google.colab import auth
from google.cloud import storage

def setup_environment(is_colab=False, bucket_name=None):
    if is_colab:
        if bucket_name is None:
            raise ValueError("bucket_name must be provided when using Google Colab with GCS")
        
        print('GOOGLE COLAB ENVIRONMENT DETECTED')
        
        # Authenticate and create a GCS client
        auth.authenticate_user()
        gcs_client = storage.Client()
        
        print(f"Attempting to access bucket: {bucket_name}")
        
        try:
            bucket = gcs_client.get_bucket(bucket_name)
        except Exception as e:
            print(f"Error accessing bucket: {str(e)}")
            raise
        
        # Create a temporary directory to mount the GCS data
        base_path = '/tmp/ecg_thesis_data'
        os.makedirs(base_path, exist_ok=True)
        
        # Download the contents of the bucket to the temporary directory
        prefix = 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/'
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            destination_path = os.path.join(base_path, blob.name)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
        
        print(f'Data downloaded from GCS bucket: {bucket_name}')
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print('LOCAL ENVIRONMENT DETECTED')
    
    print(f'Base Path: {base_path}')
    
    # Add the current directory to the Python path
    if base_path not in sys.path:
        sys.path.append(base_path)
    
    # Additional environment checks
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")
    print(f"Python path: {sys.path}")
    
    # Check for dataset existence
    dataset_path = os.path.join(base_path, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0')
    if not os.path.exists(dataset_path):
        print(f"WARNING: Dataset not found at {dataset_path}")
        print("Please ensure the CSN ECG dataset is in the correct directory.")
        print("Expected path:", dataset_path)
    else:
        print(f"Dataset found at: {dataset_path}")
    
    return base_path

def import_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing {module_name}. Make sure the file is in the correct directory.")
        sys.exit(1)

def main():
    # Setup environment and get base path
    is_colab = True  # Set this to True if you're running in Colab
    bucket_name = 'csn-ecg-dataset'  # Your GCS bucket name without any prefix
    base_path = setup_environment(is_colab, bucket_name)
    print(f"Base path set to: {base_path}")

    # Import required modules
    models = import_module('models')
    evaluation = import_module('evaluation')
    csn_ecg_data_preprocessing_colab = import_module('csn_ecg_data_preprocessing_colab')

    # Now you can use the imported modules
    build_cnn = models.build_cnn
    build_resnet18_1d = models.build_resnet18_1d
    build_resnet34_1d = models.build_resnet34_1d
    build_resnet50_1d = models.build_resnet50_1d
    build_transformer = models.build_transformer
    print_stats = evaluation.print_stats
    showConfusionMatrix = evaluation.showConfusionMatrix
    load_csn_data = csn_ecg_data_preprocessing_colab.load_data
    load_snomed_ct_mapping = csn_ecg_data_preprocessing_colab.load_snomed_ct_mapping

    # Setup parameters
    base_output_dir = os.path.join(base_path, 'output_plots')
    dataset_name = 'csn_ecg'
    model_type = 'cnn'  # Options: 'cnn', 'resnet18', 'resnet34', 'resnet50', 'transformer'

    # Create a unique output directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define base model parameters
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



    # Set up paths for the CSN ECG dataset
    database_path = os.path.join(base_path, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0')
    wfdb_dir = os.path.join(database_path, 'WFDBRecords')
    
    if not os.path.exists(wfdb_dir):
        print(f"Error: Directory {wfdb_dir} does not exist.")
        print("Current directory structure:")
        print_directory_structure(base_path)
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
    max_records = 45152
    desired_length = 1000
    print(f"Processing up to {max_records} records for CSN ECG dataset")
    csv_path = os.path.join(database_path, 'ConditionNames_SNOMED-CT.csv')

    # Define the class mapping
    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping)
    X, Y_cl = load_csn_data(wfdb_dir, data_entries, snomed_ct_mapping, max_records=max_records, desired_length=5000)

    if X.size == 0 or len(Y_cl) == 0:
        print("Error: No data was loaded. Check the data preprocessing step.")
        return

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

    # Update label mappings
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

    # Print shape of X and Y_binarized
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y_binarized: {Y_binarized.shape}")

    # Check if we have at least two samples for each class
    samples_per_class = Y_binarized.sum(axis=0)
    min_samples_per_class = samples_per_class.min()
    print(f"Samples per class: {samples_per_class}")
    print(f"Minimum samples for any class: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("Error: There are still classes with less than 2 samples.")
        return

    # Check for samples with no positive labels
    samples_with_no_labels = (Y_binarized.sum(axis=1) == 0).sum()
    if samples_with_no_labels > 0:
        print(f"Warning: {samples_with_no_labels} samples have no positive labels.")
        # Remove these samples
        valid_samples = Y_binarized.sum(axis=1) > 0
        X = X[valid_samples]
        Y_binarized = Y_binarized[valid_samples]
        print(f"Shape after removing samples with no labels - X: {X.shape}, Y_binarized: {Y_binarized.shape}")

    # Split the data into train, validation, and test sets (before scaling)
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
            **model_params
        )
    elif model_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            **model_params
        )
    elif model_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            **model_params
        )
    elif model_type == 'transformer':
        model = build_transformer(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            num_classes=num_classes,
            activation='sigmoid',  # For multi-label
            **model_params
        )
    else:
        raise ValueError("Invalid model type.")

    # Compile the model
    model.compile(
        loss='binary_crossentropy',  # For multi-label
        optimizer=keras.optimizers.Adam(learning_rate),
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
    def evaluate_model(dataset, y_true, name, output_dir, label_names):
        y_pred_prob = model.predict(dataset)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        print(f"\n{name} Performance")
        print(classification_report(y_true, y_pred, target_names=label_names))
        
        # Compute confusion matrix for each class
        conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix for each class
        for i, conf_matrix in enumerate(conf_matrices):
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {label_names[i]} - {name} Set')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name.lower()}_{label_names[i]}.png'))
            plt.close()

    # Evaluate the model on scaled data
    evaluate_model(X_train_scaled, y_train, 'Training', output_dir, label_names)
    evaluate_model(X_valid_scaled, y_valid, 'Validation', output_dir, label_names)
    evaluate_model(X_test_scaled, y_test, 'Test', output_dir, label_names)

    # Plot training history
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

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == '__main__':
    main()