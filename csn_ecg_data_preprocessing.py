# csn_ecg_data_preprocessing.py
# This script preprocesses the CSN ECG dataset, extracting SNOMED-CT codes and ECG data

import numpy as np
import os
import random
from scipy.io import loadmat
import wfdb
import logging
import pandas as pd

# Uncomment the following line to enable detailed logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_snomed_ct_mapping(csv_path):
    """
    Load SNOMED-CT codes and their corresponding full names from a CSV file.
    
    Args:
    csv_path (str): Path to the CSV file containing SNOMED-CT codes and names.
    
    Returns:
    dict: A dictionary mapping SNOMED-CT codes to their full names.
    """
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['Snomed_CT'].astype(str), df['Full Name']))
    logging.info(f"Loaded {len(mapping)} SNOMED-CT codes from CSV")
    return mapping

def extract_snomed_ct_code(header):
    """
    Extract the SNOMED-CT code from the header of an ECG record.
    
    Args:
    header (wfdb.io.record.Record): The header of an ECG record.
    
    Returns:
    str: The extracted SNOMED-CT code, or None if not found.
    """
    for comment in header.comments:
        if comment.startswith('Dx:'):
            codes = comment.split(':')[1].strip().split(',')
            return codes[-1].strip()  # Return the last code
    logging.warning(f"No Dx field found in header comments: {header.comments}")
    return None

def load_data(database_path, data_entries, snomed_ct_mapping, max_records=None):
    """
    Load and preprocess ECG data from the CSN dataset.
    
    Args:
    database_path (str): Path to the database containing ECG records.
    data_entries (list): List of record names to process.
    snomed_ct_mapping (dict): Mapping of SNOMED-CT codes to full names.
    max_records (int, optional): Maximum number of records to process.
    
    Returns:
    tuple: Numpy arrays of processed ECG data (X) and corresponding SNOMED-CT codes (Y_cl).
    """
    X, Y_cl = [], []
    processed_records = 0
    skipped_records = 0
    diagnosis_counts = {}
    no_code_count = 0
    missing_code_count = 0
    missing_codes = set()

    logging.info(f"Starting to process data. Max records: {max_records}")
    logging.info(f"Number of data entries: {len(data_entries)}")

    if max_records is not None:
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))

    for record in data_entries:
        mat_file = os.path.join(database_path, record + '.mat')
        hea_file = os.path.join(database_path, record + '.hea')

        if not os.path.exists(mat_file) or not os.path.exists(hea_file):
            logging.warning(f"Files not found for record {record}")
            skipped_records += 1
            continue

        try:
            # Load ECG data from .mat file
            mat_data = loadmat(mat_file)
            if 'val' not in mat_data:
                logging.warning(f"'val' key not found in mat file for record {record}")
                skipped_records += 1
                continue
            ecg_data = mat_data['val']

            # Read header file to get the SNOMED-CT code
            record_header = wfdb.rdheader(os.path.join(database_path, record))
            snomed_ct_code = extract_snomed_ct_code(record_header)

            if snomed_ct_code is None:
                no_code_count += 1
                logging.warning(f"No SNOMED-CT code found for record {record}")
                continue

            if snomed_ct_code not in snomed_ct_mapping:
                missing_code_count += 1
                missing_codes.add(snomed_ct_code)
                diagnosis = f"Unknown_{snomed_ct_code}"
            else:
                diagnosis = snomed_ct_mapping[snomed_ct_code]

            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1

            # Process ECG data
            X.append(ecg_data.T)  # Transpose to have shape (time_steps, leads)
            Y_cl.append(snomed_ct_code)

            processed_records += 1

            if processed_records % 100 == 0:
                logging.info(f"Processed {processed_records} records")

        except Exception as e:
            logging.error(f"Error processing record {record}: {str(e)}")
            skipped_records += 1

    # Log summary statistics
    logging.info(f"Processed {processed_records} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records with no SNOMED-CT code: {no_code_count}")
    logging.info(f"Records with missing SNOMED-CT code in mapping: {missing_code_count}")
    logging.info(f"Missing SNOMED-CT codes: {missing_codes}")
    logging.info(f"Diagnosis counts: {diagnosis_counts}")

    if len(X) == 0:
        logging.error("No data was processed successfully")
        return np.array([]), np.array([])

    return np.array(X), np.array(Y_cl)

def main():
    """
    Main function to demonstrate the usage of the data preprocessing functions.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Paths
    database_path = 'path/to/WFDBRecords'
    csv_path = 'path/to/ConditionNames_SNOMED-CT.csv'

    # Load SNOMED-CT mapping
    snomed_ct_mapping = load_snomed_ct_mapping(csv_path)

    # Get all record names
    data_entries = []
    for subdir, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.mat'):
                record_path = os.path.join(subdir, file)
                record_name = os.path.relpath(record_path, database_path)
                record_name = os.path.splitext(record_name)[0]
                data_entries.append(record_name)

    # Load data
    X, Y_cl = load_data(database_path, data_entries, snomed_ct_mapping, max_records=5000)

    # Print summary
    print(f"Loaded data shape - X: {X.shape}, Y_cl: {Y_cl.shape}")
    print(f"Unique SNOMED-CT codes: {np.unique(Y_cl)}")

if __name__ == '__main__':
    main()
