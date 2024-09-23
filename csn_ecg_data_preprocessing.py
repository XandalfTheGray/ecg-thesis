# File: csn_ecg_data_preprocessing.py
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
        random.seed(42)  # For reproducibility
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

            # Verify ECG data dimensions
            if ecg_data.ndim != 2:
                logging.warning(f"Unexpected ECG data dimensions for record {record}: {ecg_data.shape}")
                skipped_records += 1
                continue

            # Read header file to get the SNOMED-CT codes
            record_header = wfdb.rdheader(os.path.join(database_path, record))
            snomed_ct_codes = extract_snomed_ct_codes(record_header)

            if not snomed_ct_codes:
                no_code_count += 1
                logging.warning(f"No SNOMED-CT codes found for record {record}")
                continue

            # Map codes to their full names, handle unknown codes
            valid_codes = []
            for code in snomed_ct_codes:
                if code in snomed_ct_mapping:
                    valid_codes.append(code)
                    diagnosis = snomed_ct_mapping[code]
                    diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
                else:
                    valid_codes.append('Unknown')

            # Process ECG data
            X.append(ecg_data.T)  # Transpose to have shape (time_steps, leads)
            Y_cl.append(valid_codes)

            processed_records += 1

            if processed_records % 100 == 0:
                logging.info(f"Processed {processed_records} records")

        except Exception as e:
            logging.error(f"Error processing record {record}: {str(e)}")
            skipped_records += 1

    # Log summary statistics
    logging.info(f"Processed {processed_records} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records with no SNOMED-CT codes: {no_code_count}")
    logging.info(f"Records with missing SNOMED-CT codes in mapping: {missing_code_count}")
    logging.info(f"Missing SNOMED-CT codes: {missing_codes}")
    logging.info(f"Diagnosis counts: {diagnosis_counts}")

    if len(X) == 0:
        logging.error("No data was processed successfully")
        return np.array([]), np.array([])

    return np.array(X), np.array(Y_cl)