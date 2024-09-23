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

def load_snomed_ct_mapping(csv_path, selected_conditions=None):
    """
    Load SNOMED-CT codes and their corresponding full names from a CSV file.
    
    Args:
    csv_path (str): Path to the CSV file containing SNOMED-CT codes and names.
    selected_conditions (list): List of condition names to include. If None, include all.
    
    Returns:
    dict: A dictionary mapping SNOMED-CT codes to their full names.
    """
    df = pd.read_csv(csv_path)
    
    if selected_conditions:
        df = df[df['Full Name'].isin(selected_conditions)]
    
    mapping = dict(zip(df['Snomed_CT'].astype(str), df['Full Name']))
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

def plot_ecg_signal(ecg_signal, record_name, plot_dir):
    """
    Plot and save an ECG signal.
    
    Args:
    ecg_signal (np.ndarray): ECG data array with shape (time_steps, leads).
    record_name (str): Name of the ECG record.
    plot_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(15, 5))
    for lead in range(ecg_signal.shape[1]):
        plt.plot(ecg_signal[:, lead], label=f'Lead {lead+1}')
    plt.title(f'ECG Signal for Record: {record_name}')
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

def load_data(database_path, data_entries, snomed_ct_mapping, max_records=None, desired_length=5000, num_plots=5, plot_dir='output_plots/test_ecg_plots/', batch_size=1000):
    X, Y_cl = [], []
    processed_records = 0
    skipped_records = 0
    diagnosis_counts = {}
    no_code_count = 0
    missing_code_count = 0
    missing_codes = set()
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    logging.info(f"Starting to process data. Max records: {max_records}")
    logging.info(f"Number of data entries: {len(data_entries)}")
    
    if max_records is not None:
        random.seed(42)
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
    
    for i in range(0, len(data_entries), batch_size):
        batch = data_entries[i:i+batch_size]
        for record in batch:
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
        
                valid_codes = []
                for code in snomed_ct_codes:
                    if code in snomed_ct_mapping:
                        valid_codes.append(code)
                        diagnosis = snomed_ct_mapping[code]
                        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
                    else:
                        valid_codes.append('Unknown')
                        diagnosis_counts['Unknown'] = diagnosis_counts.get('Unknown', 0) + 1
                        missing_code_count += 1
                        missing_codes.add(code)
        
                ecg_padded = pad_ecg_data(ecg_data.T, desired_length)
                X.append(ecg_padded)
                Y_cl.append(valid_codes)
        
                if processed_records < num_plots:
                    plot_ecg_signal(ecg_padded, record, plot_dir)
                    logging.info(f"Plotted ECG signal for record {record}")
        
                processed_records += 1
        
                if processed_records % 100 == 0:
                    logging.info(f"Processed {processed_records} records")
        
                if max_records and processed_records >= max_records:
                    break
        
            except Exception as e:
                logging.error(f"Error processing record {record}: {str(e)}")
                skipped_records += 1
        
        if max_records and processed_records >= max_records:
            break
    
    logging.info(f"Processed {processed_records} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records with no SNOMED-CT codes: {no_code_count}")
    logging.info(f"Records with missing SNOMED-CT codes in mapping: {missing_code_count}")
    logging.info(f"Missing SNOMED-CT codes: {missing_codes}")
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

    # Specify the conditions you want to classify
    selected_conditions = [
        'Sinus Rhythm',
        'Atrial Fibrillation',
        'Atrial Flutter',
        'Sinus Bradycardia',
        'Sinus Tachycardia'
    ]

    # Load SNOMED-CT mapping with selected conditions
    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, selected_conditions)

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
        num_plots=5,
        plot_dir='output_plots/test_ecg_plots/'
    )

    # Print summary
    print(f"Loaded data shape - X: {X.shape}, Y_cl: {len(Y_cl)}")
    unique_codes = set(code for sublist in Y_cl for code in sublist)
    print(f"Unique SNOMED-CT codes: {unique_codes}")

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
