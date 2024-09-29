# csn_ecg_data_preprocessing_colab.py
# This script preprocesses the CSN ECG dataset for Colab, extracting SNOMED-CT codes and ECG data

import numpy as np
import os
import random
from scipy.io import loadmat
import wfdb
import logging
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from google.cloud import storage
import io
from tqdm import tqdm
import concurrent.futures
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

# Uncomment for detailed logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_snomed_ct_mapping(csv_path, class_mapping, bucket=None):
    """
    Load SNOMED-CT codes and their corresponding class names from a CSV file.
    
    Args:
    csv_path (str): Path to the CSV file containing SNOMED-CT codes and names.
    class_mapping (dict): Dictionary mapping class names to lists of condition names.
    bucket (google.cloud.storage.bucket.Bucket, optional): GCS bucket object.
    
    Returns:
    dict: A dictionary mapping SNOMED-CT codes to their corresponding class names.
    """
    try:
        if bucket:
            blob = bucket.blob(csv_path)
            content = blob.download_as_text()
            df = pd.read_csv(io.StringIO(content))
            print(f"Downloaded and read CSV from GCS: {csv_path}")
        else:
            df = pd.read_csv(csv_path)
            print(f"Read CSV from local path: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to load CSV file at {csv_path}: {e}")
        raise
    
    condition_to_class = {condition.lower(): class_name 
                          for class_name, conditions in class_mapping.items() 
                          for condition in conditions}
    
    mapping = {str(row['Snomed_CT']): condition_to_class.get(row['Full Name'].lower(), 'Other') 
               for _, row in df.iterrows()}
    
    logging.info(f"Loaded {len(mapping)} SNOMED-CT codes from CSV")
    return mapping

def extract_snomed_ct_codes(header_content):
    """
    Extract all SNOMED-CT codes from the header content of an ECG record.
    
    Args:
    header_content (str): The content of the header file.
    
    Returns:
    list: A list of extracted SNOMED-CT codes, or an empty list if none found.
    """
    for line in header_content.split('\n'):
        if line.startswith('Dx:'):
            return [code.strip() for code in line.split(':')[1].strip().split(',')]
    logging.warning(f"No Dx field found in header content")
    return []

def pad_ecg_data(ecg_data, desired_length):
    """
    Pad or truncate ECG data to a fixed number of time steps.
    
    Args:
    ecg_data (np.ndarray): ECG data array with shape (time_steps, leads).
    desired_length (int): The desired number of time steps.
    
    Returns:
    np.ndarray: ECG data array with shape (desired_length, leads).
    """
    current_length = ecg_data.shape[0]
    if current_length < desired_length:
        return np.pad(ecg_data, ((0, desired_length - current_length), (0, 0)), mode='constant')
    return ecg_data[:desired_length, :]

def plot_ecg_signal(ecg_signal, record_name, plot_dir, class_names):
    """
    Plot and save an ECG signal.
    
    Args:
    ecg_signal (np.ndarray): ECG data array with shape (time_steps, leads).
    record_name (str): Name of the ECG record.
    plot_dir (str): Directory where the plot will be saved.
    class_names (list): List of class names associated with the ECG signal.
    """
    plt.figure(figsize=(15, 5))
    for lead in range(ecg_signal.shape[1]):
        plt.plot(ecg_signal[:, lead], label=f'Lead {lead+1}')
    plt.title(f'ECG Signal for Record: {record_name} | Classes: {", ".join(class_names)}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    simplified_name = record_name.replace('\\', '_').replace('/', '_')
    plt.savefig(os.path.join(plot_dir, f'{simplified_name}.png'))
    plt.close()

def process_record(record, database_path, snomed_ct_mapping, desired_length, plot_dir, num_plots_per_class, bucket):
    """
    Process a single ECG record.
    
    Args:
    record (str): Relative path of the ECG record within WFDBRecords (e.g., '01/010/JS00001').
    database_path (str): Path to the database containing ECG records.
    snomed_ct_mapping (dict): Mapping of SNOMED-CT codes to class names.
    desired_length (int): The desired number of time steps for ECG data.
    plot_dir (str): Directory where ECG plots will be saved.
    num_plots_per_class (int): Number of ECG signals to plot per class.
    bucket (google.cloud.storage.bucket.Bucket): GCS bucket object.
    
    Returns:
    tuple: Processed ECG data, valid classes, and error message (if any).
    """
    mat_file = f'{database_path}/{record}.mat'
    hea_file = f'{database_path}/{record}.hea'

    if not bucket:
        # Local file system access (if bucket is None)
        if not os.path.exists(mat_file) or not os.path.exists(hea_file):
            return None, None, f"Files not found for record {record}"

        try:
            mat_data = loadmat(mat_file)
            if 'val' not in mat_data:
                return None, None, f"'val' key not found in mat file for record {record}"
            ecg_data = mat_data['val']

            if ecg_data.ndim != 2:
                return None, None, f"Unexpected ECG data dimensions for record {record}: {ecg_data.shape}"

            record_header = wfdb.rdheader(os.path.join(database_path, record))
            snomed_ct_codes = extract_snomed_ct_codes(record_header)

            if not snomed_ct_codes:
                return None, None, f"No SNOMED-CT codes found for record {record}"

            valid_classes = list(set(snomed_ct_mapping.get(code, 'Other') for code in snomed_ct_codes))
            if len(valid_classes) > 1 and 'Other' in valid_classes:
                valid_classes.remove('Other')

            ecg_padded = pad_ecg_data(ecg_data.T, desired_length)

            # Plot ECG signals per class (if needed)
            if num_plots_per_class > 0:
                plot_ecg_signal(ecg_padded, record, plot_dir, valid_classes)

            return ecg_padded, valid_classes, None
        except Exception as e:
            return None, None, f"Error processing record {record}: {str(e)}"
    else:
        # GCS access
        try:
            # Construct blob paths
            mat_blob = bucket.blob(mat_file)
            hea_blob = bucket.blob(hea_file)

            if not mat_blob.exists():
                raise FileNotFoundError(f"Mat file not found: {mat_file}")
            if not hea_blob.exists():
                raise FileNotFoundError(f"Hea file not found: {hea_file}")

            # Download content
            mat_content = mat_blob.download_as_bytes()
            hea_content = hea_blob.download_as_string().decode('utf-8')

            # Load mat file
            mat_data = loadmat(io.BytesIO(mat_content))
            ecg_data = mat_data['val']

            if ecg_data.ndim != 2:
                return None, None, f"Unexpected ECG data dimensions for record {record}: {ecg_data.shape}"

            # Process header content
            snomed_ct_codes = extract_snomed_ct_codes(hea_content)

            if not snomed_ct_codes:
                return None, None, f"No SNOMED-CT codes found for record {record}"

            valid_classes = list(set(snomed_ct_mapping.get(code, 'Other') for code in snomed_ct_codes))
            if len(valid_classes) > 1 and 'Other' in valid_classes:
                valid_classes.remove('Other')

            # Pad or truncate ECG data
            ecg_padded = pad_ecg_data(ecg_data.T, desired_length)

            # Plot ECG signals per class (if needed)
            if num_plots_per_class > 0:
                plot_ecg_signal(ecg_padded, record, plot_dir, valid_classes)

            return ecg_padded, valid_classes, None
        except Exception as e:
            return None, None, f"Error processing record {record}: {str(e)}"

def load_csn_data(base_path, data_entries, snomed_ct_mapping, max_records=None, desired_length=5000, plot_dir='plots', num_plots_per_class=0, bucket=None):
    X, Y_cl = [], []
    client = storage.Client()
    if bucket is None:
        bucket_obj = None
    else:
        bucket_obj = client.get_bucket(bucket)
    
    logging.info(f"Starting to load data from {len(data_entries)} records")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for record in data_entries[:max_records]:
            futures.append(executor.submit(
                process_record, 
                record, 
                base_path, 
                snomed_ct_mapping, 
                desired_length, 
                plot_dir, 
                num_plots_per_class, 
                bucket_obj
            ))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Loading records"):
            ecg_padded, valid_classes, error = future.result()
            if ecg_padded is not None and valid_classes:
                X.append(ecg_padded)
                Y_cl.append(valid_classes)
            elif error:
                logging.error(error)
    
    logging.info(f"Loaded {len(X)} records successfully")
    
    if len(X) == 0:
        logging.error("No records were successfully loaded. Check the data files and paths.")
        return None, None
    
    logging.info(f"Final X shape: {np.array(X).shape}, Y_cl length: {len(Y_cl)}")
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y_cl)
    
    # Create a MultiLabelBinarizer to convert Y_cl to one-hot encoded format
    mlb = MultiLabelBinarizer()
    Y_encoded = mlb.fit_transform(Y)
    
    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, Y_encoded))
    
    # Use cache and prefetch to optimize memory usage
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    
    return dataset, mlb.classes_

def find_mat_files(directory):
    """
    Recursively find all .mat files in the given directory and its subdirectories.
    
    Args:
    directory (str): The directory to search in.
    
    Returns:
    list: A list of paths to .mat files relative to the directory.
    """
    mat_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat'):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                record_name = os.path.splitext(relative_path)[0]  # Remove '.mat'
                mat_files.append(record_name)
    return mat_files

def main():
    """
    Main function to demonstrate the usage of the data preprocessing functions.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup environment
    is_colab = True  # Set to True if running in Colab
    bucket_name = 'csn-ecg-dataset'  # Your GCS bucket name

    if is_colab:
        from google.colab import auth
        auth.authenticate_user()
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        print('GOOGLE COLAB ENVIRONMENT DETECTED')
    else:
        bucket = None
        print('LOCAL ENVIRONMENT DETECTED')

    # Set the base path within the bucket
    base_path = 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
    database_path = f'{base_path}/WFDBRecords'
    csv_path = f'{base_path}/ConditionNames_SNOMED-CT.csv'

    # Check if paths exist (for local access)
    if not is_colab:
        if not os.path.exists(base_path):
            logging.error(f"Base path does not exist: {base_path}")
            return
        if not os.path.exists(database_path):
            logging.error(f"Database path does not exist: {database_path}")
            return
        if not os.path.exists(csv_path):
            logging.error(f"CSV file does not exist: {csv_path}")
            return

    print(f"Base path: {base_path}")
    print(f"Database path: {database_path}")
    print(f"CSV path: {csv_path}")

    # List contents of the base path (for debugging)
    if is_colab and bucket:
        print(f"Listing first 10 contents of {database_path}:")
        blobs = list(bucket.list_blobs(prefix=database_path, max_results=10))
        for blob in blobs:
            print(f" - {blob.name}")
    else:
        print(f"Contents of base path:")
        for item in os.listdir(base_path):
            print(f"  {item}")

    # Find all .mat files
    if is_colab and bucket:
        mat_files = list(bucket.list_blobs(prefix=database_path))
        mat_records = []
        for blob in mat_files:
            if blob.name.endswith('.mat'):
                relative_path = os.path.relpath(blob.name, database_path)
                record_name = os.path.splitext(relative_path)[0]  # Remove '.mat'
                mat_records.append(record_name)
        print(f"Found {len(mat_records)} .mat files in GCS.")
    else:
        mat_records = find_mat_files(database_path)
        print(f"Found {len(mat_records)} .mat files in local directory.")

    if len(mat_records) == 0:
        logging.error("No .mat files found. Check the database path and file structure.")
        return

    # Define class mapping
    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 
                 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    # Load SNOMED-CT mapping
    try:
        snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping, bucket=bucket)
        print(f"Loaded {len(snomed_ct_mapping)} SNOMED-CT codes.")
    except Exception as e:
        logging.error(f"Failed to load SNOMED-CT mapping: {e}")
        return

    # Set preprocessing parameters
    max_records = 10  # Adjust as needed
    desired_length = 1000  # Adjust based on your requirements
    plot_dir = 'output_plots'  # Directory to save plots
    num_plots_per_class = 0  # Set >0 to enable plotting

    # Load and preprocess data
    try:
        dataset, classes = load_csn_data(
            base_path=database_path,
            data_entries=mat_records,
            snomed_ct_mapping=snomed_ct_mapping,
            max_records=max_records,
            desired_length=desired_length,
            plot_dir=plot_dir,
            num_plots_per_class=num_plots_per_class,
            bucket=bucket_name if is_colab else None
        )
        
        if dataset is None or classes is None:
            logging.error("Failed to load and preprocess ECG data.")
            return

        print(f"Loaded dataset shape - X: {dataset.element_spec[0].shape}, Y_cl: {dataset.element_spec[1].shape}")
        print(f"Unique classes: {classes}")

        # Example: Iterate through a batch
        for X_batch, Y_batch in dataset.take(1):
            print(f"Batch X shape: {X_batch.shape}")
            print(f"Batch Y shape: {Y_batch.shape}")
        
    except Exception as e:
        logging.error(f"Error during data loading or preprocessing: {e}")
        return

    print("Data preprocessing completed successfully.")

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
