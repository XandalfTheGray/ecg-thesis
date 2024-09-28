# csn_ecg_data_preprocessing.py
# This script preprocesses the CSN ECG dataset, extracting SNOMED-CT codes and ECG data

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

# Uncomment for detailed logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_snomed_ct_mapping(csv_path, class_mapping):
    """
    Load SNOMED-CT codes and their corresponding class names from a CSV file.
    
    Args:
    csv_path (str): Path to the CSV file containing SNOMED-CT codes and names.
    class_mapping (dict): Dictionary mapping class names to lists of condition names.
    
    Returns:
    dict: A dictionary mapping SNOMED-CT codes to their corresponding class names.
    """
    df = pd.read_csv(csv_path)
    
    condition_to_class = {condition.lower(): class_name 
                          for class_name, conditions in class_mapping.items() 
                          for condition in conditions}
    
    mapping = {str(row['Snomed_CT']): condition_to_class.get(row['Full Name'].lower(), 'Other') 
               for _, row in df.iterrows()}
    
    logging.info(f"Loaded {len(mapping)} SNOMED-CT codes from CSV")
    return mapping

def extract_snomed_ct_codes(header):
    """
    Extract all SNOMED-CT codes from the header of an ECG record.
    
    Args:
    header (wfdb.io.record.Record): The header of an ECG record.
    
    Returns:
    list: A list of extracted SNOMED-CT codes, or an empty list if none found.
    """
    for comment in header.comments:
        if comment.startswith('Dx:'):
            return [code.strip() for code in comment.split(':')[1].strip().split(',')]
    logging.warning(f"No Dx field found in header comments: {header.comments}")
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

def load_data(database_path, data_entries, snomed_ct_mapping, max_records=None, desired_length=5000, num_plots_per_class=1, plot_dir='output_plots/class_ecg_plots/'):
    """
    Load and preprocess ECG data from the CSN dataset.
    
    Args:
    database_path (str): Path to the database containing ECG records.
    data_entries (list): List of record names to process.
    snomed_ct_mapping (dict): Mapping of SNOMED-CT codes to class names.
    max_records (int, optional): Maximum number of records to process.
    desired_length (int): The fixed number of time steps for ECG data.
    num_plots_per_class (int): Number of ECG signals to plot per class.
    plot_dir (str): Directory where class-specific ECG plots will be saved.
    
    Returns:
    tuple: Numpy array of processed ECG data (X) and list of corresponding class names (Y_cl).
    """
    if not data_entries:
        logging.error("No data entries found. Check the database path and file extensions.")
        return np.array([]), []

    if max_records is not None:
        random.seed(42)
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))

    logging.info(f"Processing {len(data_entries)} records")

    start_time = time.time()
    
    # Use multiprocessing to process records in parallel
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        process_func = partial(process_record, database_path=database_path, snomed_ct_mapping=snomed_ct_mapping, 
                               desired_length=desired_length, plot_dir=plot_dir, num_plots_per_class=num_plots_per_class)
        results = pool.map(process_func, data_entries)

    X, Y_cl = [], []
    diagnosis_counts = {}
    skipped_records = 0
    no_code_count = 0

    for ecg_padded, valid_classes, error_message in results:
        if ecg_padded is not None and valid_classes:
            X.append(ecg_padded)
            Y_cl.append(valid_classes)
            for cls in valid_classes:
                diagnosis_counts[cls] = diagnosis_counts.get(cls, 0) + 1
        elif error_message:
            if "No SNOMED-CT codes found" in error_message:
                no_code_count += 1
            else:
                skipped_records += 1
            logging.warning(error_message)

    end_time = time.time()
    logging.info(f"Processing time: {end_time - start_time:.2f} seconds")
    logging.info(f"Processed {len(X)} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records with no SNOMED-CT codes: {no_code_count}")
    logging.info(f"Diagnosis counts: {diagnosis_counts}")
    
    return np.array(X), Y_cl

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

    X, Y_cl = load_data(
        database_path, 
        data_entries, 
        snomed_ct_mapping, 
        max_records=20000, 
        desired_length=1000, 
        num_plots_per_class=1,
        plot_dir='output_plots/class_ecg_plots/'
    )

    print(f"Loaded data shape - X: {X.shape}, Y_cl: {len(Y_cl)}")
    unique_classes = set(class_name for sublist in Y_cl for class_name in sublist)
    print(f"Unique classes: {unique_classes}")

    if len(X) > 0:
        plt.figure(figsize=(15, 5))
        for lead in range(X[0].shape[1]):
            plt.plot(X[0][:, lead], label=f'Lead {lead+1}')
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