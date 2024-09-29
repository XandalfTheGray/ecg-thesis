# File: csn_ecg_train_model_colab.py
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
from google.colab import auth, drive
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tensorflow as tf
import importlib
import logging

def download_blob(blob, base_path):
    destination_path = os.path.join(base_path, blob.name)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    blob.download_to_filename(destination_path)

def setup_environment(is_colab=False, bucket_name=None):
    if is_colab:
        if bucket_name is None:
            raise ValueError("bucket_name must be provided when using Google Colab with GCS")
        
        print('GOOGLE COLAB ENVIRONMENT DETECTED')
        
        # Authenticate
        auth.authenticate_user()
        
        # Create a client
        client = storage.Client()
        
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        
        print(f"Accessing GCS bucket: {bucket_name}")
        
        # List files in the bucket to verify access
        blobs = list(bucket.list_blobs(prefix='a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/', max_results=10))
        
        if not blobs:
            print(f"WARNING: No files found in the specified path in the GCS bucket.")
        else:
            print(f"Successfully accessed the GCS bucket. Found files in the specified path.")
        
        base_path = f'gs://{bucket_name}'
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print('LOCAL ENVIRONMENT DETECTED')
        bucket = None
    
    print(f'Base Path: {base_path}')
    
    # Additional environment checks
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    return base_path, bucket

def import_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Error importing {module_name}. Please make sure it's installed.")
        return None

def create_dataset(X, y, batch_size=32, shuffle=True, prefetch=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup environment and get base path
    is_colab = True  # Set this to True if you're running in Colab
    bucket_name = 'csn-ecg-dataset'  # Your GCS bucket name
    base_path, bucket = setup_environment(is_colab, bucket_name)
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
    load_csn_data = csn_ecg_data_preprocessing_colab.load_csn_data
    load_snomed_ct_mapping = csn_ecg_data_preprocessing_colab.load_snomed_ct_mapping

    # Setup parameters
    base_output_dir = os.path.join('output_plots')
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
    database_path = 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
    csv_path = f'{database_path}/ConditionNames_SNOMED-CT.csv'
    
    # Gather all record names
    data_entries = []
    if bucket:
        prefix = f'{database_path}/WFDBRecords/'
        blobs = list(bucket.list_blobs(prefix=prefix))
        data_entries = [blob.name.split('/')[-1].split('.')[0] for blob in blobs if blob.name.endswith('.mat')]
    else:
        for subdir, dirs, files in os.walk(database_path):
            data_entries.extend([os.path.join(os.path.relpath(subdir, database_path), os.path.splitext(file)[0]) 
                                 for file in files if file.endswith('.mat')])

    print(f"Total records found for CSN ECG: {len(data_entries)}")
    
    if len(data_entries) == 0:
        print("Error: No records found. Check the database path and file structure.")
        return

    # Load and preprocess data
    max_records = 50000  # Increased from 5000 to 50000
    desired_length = 1000
    print(f"Processing up to {max_records} records for CSN ECG dataset")

    # Define the class mapping
    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    try:
        print("Loading SNOMED-CT mapping...")
        snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping, bucket=bucket)
        print("SNOMED-CT mapping loaded successfully")
        print(f"Number of SNOMED-CT codes loaded: {len(snomed_ct_mapping)}")
        
        print("Starting to load and preprocess ECG data...")
        dataset = load_csn_data(base_path, data_entries, snomed_ct_mapping, max_records=max_records, desired_length=desired_length, bucket=bucket)
        print("ECG data loading and preprocessing completed")
        
        # Split the dataset into train, validation, and test sets
        train_size = int(0.7 * max_records)
        val_size = int(0.15 * max_records)
        test_size = max_records - train_size - val_size

        train_dataset = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
        val_dataset = dataset.skip(train_size).take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = dataset.skip(train_size + val_size).take(test_size).batch(32).prefetch(tf.data.AUTOTUNE)

        # Get input shape and number of classes from the dataset
        for batch in train_dataset.take(1):
            input_shape = batch[0].shape[1:]
            num_classes = batch[1].shape[1]
            break

        # Build the neural network model
        if model_type == 'cnn':
            model = build_cnn(
                input_shape=input_shape,
                num_classes=num_classes,
                activation='sigmoid',  # For multi-label
                **model_params
            )
        # ... (keep other model types)

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

        # Train the model using the dataset
        history = model.fit(
            train_dataset,
            epochs=30,
            validation_data=val_dataset,
            callbacks=callbacks,
        )

        # Define evaluation function
        def evaluate_model(dataset, name, output_dir, label_names):
            y_pred_prob = model.predict(dataset)
            y_true = np.concatenate([y for x, y in dataset], axis=0)
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

        # Evaluate the model on the datasets
        evaluate_model(train_dataset, 'Training', output_dir, label_names)
        evaluate_model(val_dataset, 'Validation', output_dir, label_names)
        evaluate_model(test_dataset, 'Test', output_dir, label_names)

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

        # ... (keep the rest of the main function)

    except Exception as e:
        print(f"Error during data loading or model training: {str(e)}")
        logging.exception("Exception occurred during data loading or model training")
        return

    # ... (keep the rest of the main function)

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