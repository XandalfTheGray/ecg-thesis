# csn_ecg_data_preprocessing.py
# This script preprocesses the CSN ECG dataset, extracting SNOMED-CT codes and ECG data

import numpy as np
import os
import random
from scipy.io import loadmat
import wfdb
import logging
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Uncomment the following line to enable detailed logging
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
    
    # Create a reverse mapping from condition names to class names
    condition_to_class = {}
    for class_name, conditions in class_mapping.items():
        for condition in conditions:
            condition_to_class[condition.lower()] = class_name
    
    mapping = {}
    for _, row in df.iterrows():
        snomed_ct_code = str(row['Snomed_CT'])
        condition_name = row['Full Name'].lower()
        if condition_name in condition_to_class:
            mapping[snomed_ct_code] = condition_to_class[condition_name]
    
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
    codes = []
    for comment in header.comments:
        if comment.startswith('Dx:'):
            codes_part = comment.split(':')[1].strip()
            codes = [code.strip() for code in codes_part.split(',')]
            break
    if not codes:
        logging.warning(f"No Dx field found in header comments: {header.comments}")
    return codes

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
        padding = np.zeros((desired_length - current_length, ecg_data.shape[1]))
        ecg_padded = np.vstack((ecg_data, padding))
    else:
        ecg_padded = ecg_data[:desired_length, :]
    return ecg_padded

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
    class_label = ', '.join(class_names)
    plt.title(f'ECG Signal for Record: {record_name} | Classes: {class_label}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.tight_layout()
    
    # Create the plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot with a simplified filename
    simplified_name = record_name.replace('\\', '_').replace('/', '_')
    plt.savefig(os.path.join(plot_dir, f'{simplified_name}.png'))
    plt.close()

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
    X, Y_cl = [], []
    processed_records = 0
    skipped_records = 0
    diagnosis_counts = {}
    no_code_count = 0
    plotted_classes = set()
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    logging.info(f"Starting to process data. Max records: {max_records}")
    logging.info(f"Number of data entries: {len(data_entries)}")
    
    if max_records is not None:
        random.seed(42)
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
    
    for i, record in enumerate(data_entries):
        mat_file = os.path.join(database_path, record + '.mat')
        hea_file = os.path.join(database_path, record + '.hea')
    
        if not os.path.exists(mat_file) or not os.path.exists(hea_file):
            logging.warning(f"Files not found for record {record}")
            skipped_records += 1
            continue
    
        try:
            mat_data = loadmat(mat_file)
            if 'val' not in mat_data:
                logging.warning(f"'val' key not found in mat file for record {record}")
                skipped_records += 1
                continue
            ecg_data = mat_data['val']
    
            if ecg_data.ndim != 2:
                logging.warning(f"Unexpected ECG data dimensions for record {record}: {ecg_data.shape}")
                skipped_records += 1
                continue
    
            record_header = wfdb.rdheader(os.path.join(database_path, record))
            snomed_ct_codes = extract_snomed_ct_codes(record_header)
    
            if not snomed_ct_codes:
                no_code_count += 1
                logging.warning(f"No SNOMED-CT codes found for record {record}")
                continue
    
            valid_classes = []
            for code in snomed_ct_codes:
                if code in snomed_ct_mapping:
                    class_name = snomed_ct_mapping[code]
                    valid_classes.append(class_name)
                    diagnosis_counts[class_name] = diagnosis_counts.get(class_name, 0) + 1
                else:
                    valid_classes.append('Other')
                    diagnosis_counts['Other'] = diagnosis_counts.get('Other', 0) + 1
    
            # Remove duplicates and 'Other' if there are other valid classes
            valid_classes = list(set(valid_classes))
            if len(valid_classes) > 1 and 'Other' in valid_classes:
                valid_classes.remove('Other')
    
            ecg_padded = pad_ecg_data(ecg_data.T, desired_length)
            X.append(ecg_padded)
            Y_cl.append(valid_classes)
    
            # Plot ECG signals per class
            for cls in valid_classes:
                if cls not in plotted_classes and list(Y_cl).count(cls) <= num_plots_per_class:
                    plot_ecg_signal(ecg_padded, record, plot_dir, valid_classes)
                    logging.info(f"Plotted ECG signal for class {cls} in record {record}")
                    plotted_classes.add(cls)
    
            processed_records += 1
    
            if processed_records % 100 == 0:
                logging.info(f"Processed {processed_records} records")
    
            if max_records and processed_records >= max_records:
                break
    
        except Exception as e:
            logging.error(f"Error processing record {record}: {str(e)}")
            skipped_records += 1
    
    logging.info(f"Processed {processed_records} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records with no SNOMED-CT codes: {no_code_count}")
    logging.info(f"Diagnosis counts: {diagnosis_counts}")
    
    if len(X) == 0:
        logging.error("No data was processed successfully")
        return np.array([]), []
    
    return np.array(X), Y_cl

def main():
    """
    Main function to demonstrate the usage of the data preprocessing functions.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Paths
    database_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 
                                 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                                 'WFDBRecords')
    csv_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 
                            'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                            'ConditionNames_SNOMED-CT.csv')

    # Define the class mapping
    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 
                 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    # Load SNOMED-CT mapping with class mapping
    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping)

    # Get all record names
    data_entries = []
    for subdir, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.mat'):
                record_path = os.path.join(subdir, file)
                record_name = os.path.relpath(record_path, database_path)
                record_name = os.path.splitext(record_name)[0]
                data_entries.append(record_name)

    # Load data with plotting
    X, Y_cl = load_data(
        database_path, 
        data_entries, 
        snomed_ct_mapping, 
        max_records=5000, 
        desired_length=5000, 
        num_plots_per_class=1,
        plot_dir='output_plots/class_ecg_plots/'
    )

    # Print summary
    print(f"Loaded data shape - X: {X.shape}, Y_cl: {len(Y_cl)}")
    unique_classes = set(class_name for sublist in Y_cl for class_name in sublist)
    print(f"Unique classes: {unique_classes}")

    # Display a sample plot
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

if __name__ == '__main__':
    main()
