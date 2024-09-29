# csn_ecg_data_preprocessing_colab.py
# This script preprocesses the CSN ECG dataset for colab, extracting SNOMED-CT codes and ECG data

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
    if bucket:
        blob = bucket.blob(csv_path)
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
    else:
        df = pd.read_csv(csv_path)
    
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

def process_record(record, database_path, snomed_ct_mapping, desired_length, plot_dir, num_plots_per_class):
    """
    Process a single ECG record.
    
    Args:
    record (str): Name of the ECG record.
    database_path (str): Path to the database containing ECG records.
    snomed_ct_mapping (dict): Mapping of SNOMED-CT codes to class names.
    desired_length (int): The desired number of time steps for ECG data.
    plot_dir (str): Directory where ECG plots will be saved.
    num_plots_per_class (int): Number of ECG signals to plot per class.
    
    Returns:
    tuple: Processed ECG data, valid classes, and error message (if any).
    """
    mat_file = os.path.join(database_path, record + '.mat')
    hea_file = os.path.join(database_path, record + '.hea')

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

def load_csn_data(base_path, data_entries, snomed_ct_mapping, max_records=None, desired_length=5000, bucket=None):
    X, Y_cl = [], []
    client = storage.Client()
    bucket = client.get_bucket(base_path.replace('gs://', ''))
    
    logging.info(f"Starting to load data from {len(data_entries)} records")
    
    # Function to process a single record
    def process_record(record):
        # Construct the correct path for .mat and .hea files
        record_path = f'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/{record[:2]}/{record[:3]}/{record}'
        mat_file = f'{record_path}.mat'
        hea_file = f'{record_path}.hea'

        print(f"Attempting to access: {mat_file}")  # Add this line

        try:
            # Download files from GCS
            mat_blob = bucket.blob(mat_file)
            hea_blob = bucket.blob(hea_file)

            print(f"Mat blob: {mat_blob.name}")  # Add this line
            print(f"Hea blob: {hea_blob.name}")  # Add this line

            mat_content = mat_blob.download_as_bytes()
            hea_content = hea_blob.download_as_string().decode('utf-8')

            # Load mat file
            mat_data = loadmat(io.BytesIO(mat_content))
            ecg_data = mat_data['val']

            # Process header file
            snomed_ct_codes = extract_snomed_ct_codes(hea_content)
            valid_classes = list(set(snomed_ct_mapping.get(code, 'Other') for code in snomed_ct_codes))
            if len(valid_classes) > 1 and 'Other' in valid_classes:
                valid_classes.remove('Other')

            # Pad or truncate ECG data
            ecg_padded = pad_ecg_data(ecg_data.T, desired_length)

            return ecg_padded, valid_classes
        except Exception as e:
            logging.error(f"Error processing record {record}: {str(e)}")
            return None, None

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_record = {executor.submit(process_record, record): record for record in data_entries[:max_records]}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_record), total=len(future_to_record), desc="Loading records"):
            record = future_to_record[future]
            try:
                ecg_padded, valid_classes = future.result()
                if ecg_padded is not None and valid_classes:
                    X.append(ecg_padded)
                    Y_cl.append(valid_classes)
            except Exception as exc:
                logging.error(f'{record} generated an exception: {exc}')

    logging.info(f"Loaded {len(X)} records successfully")
    
    if len(X) == 0:
        logging.error("No records were successfully loaded. Check the data files and paths.")
        return None

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
    list: A list of paths to .mat files.
    """
    mat_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))
    return mat_files

def main():
    """
    Main function to demonstrate the usage of the data preprocessing functions.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Paths
    base_path = 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
    database_path = os.path.join(base_path, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 'WFDBRecords')
    csv_path = os.path.join(base_path, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 'ConditionNames_SNOMED-CT.csv')

    # Check if paths exist
    if not os.path.exists(base_path):
        logging.error(f"Base path does not exist: {base_path}")
        return
    if not os.path.exists(database_path):
        logging.error(f"Database path does not exist: {database_path}")
        return
    if not os.path.exists(csv_path):
        logging.error(f"CSV file does not exist: {csv_path}")
        return

    logging.info(f"Base path: {base_path}")
    logging.info(f"Database path: {database_path}")
    logging.info(f"CSV path: {csv_path}")

    # List contents of the base path
    logging.info(f"Contents of base path:")
    for item in os.listdir(base_path):
        logging.info(f"  {item}")

    # List contents of the database path
    logging.info(f"Contents of database path:")
    for item in os.listdir(database_path):
        logging.info(f"  {item}")

    # Find .mat files recursively
    mat_files = find_mat_files(database_path)
    logging.info(f"Found {len(mat_files)} .mat files in the database directory and its subdirectories")

    if not mat_files:
        logging.error("No .mat files found in the database directory or its subdirectories")
        return

    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 
                 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping)

    # Use the full paths of .mat files, but remove the database_path prefix and .mat extension
    data_entries = [os.path.relpath(file, database_path)[:-4] for file in mat_files]
    logging.info(f"Prepared {len(data_entries)} data entries for processing")

    dataset, classes = load_csn_data(
        base_path, 
        data_entries, 
        snomed_ct_mapping, 
        max_records=20000, 
        desired_length=1000, 
        bucket=None
    )

    print(f"Loaded dataset shape - X: {dataset.element_spec[0].shape}, Y_cl: {dataset.element_spec[1].shape}")
    print(f"Unique classes: {classes}")

    if len(dataset) > 0:
        plt.figure(figsize=(15, 5))
        for lead in range(dataset.element_spec[0].shape[1]):
            plt.plot(dataset.element_spec[0][:, lead], label=f'Lead {lead+1}')
        plt.title(f'Sample ECG Signal')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', ncol=4, fontsize='small')
        plt.tight_layout()
        plt.show()
    else:
        logging.warning("No data was processed successfully. Check the logs for errors.")

if __name__ == '__main__':
    main()