# csnecg_data_preprocessing.py

import os
import random
import numpy as np
import wfdb
import logging
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import zipfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_snomed_ct_mapping(csv_path, class_mapping):
    """
    Load SNOMED-CT codes and their corresponding class names from a CSV file.
    """
    df = pd.read_csv(csv_path)
    
    # Create a reverse mapping from condition names to class names
    condition_to_class = {}
    for class_name, conditions in class_mapping.items():
        for condition in conditions:
            condition_to_class[condition.lower()] = class_name
    
    # Create the final mapping from SNOMED-CT codes to class names
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

def filter_ecg(signal, fs=500, lowcut=0.5, highcut=50.0):
    """
    Apply bandpass filter to ECG signal.
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg_data, fs=500):
    """
    Detect R-peaks in the ECG signal using multiple leads for robust detection.
    """
    # Use leads I, II, and V1-V6 for peak detection
    important_leads = [0, 1, 6, 7, 8, 9, 10, 11]

    all_peaks = []
    for lead_idx in important_leads:
        # Filter the signal
        filtered_signal = filter_ecg(ecg_data[:, lead_idx], fs=fs)
        
        # Find peaks in filtered signal
        peaks, _ = find_peaks(filtered_signal, 
                              distance=50,
                              height=0.1,  # Minimum height for peaks
                              prominence=0.2)  # Minimum prominence for peaks
        all_peaks.append(peaks)
    
    # Find consensus peaks
    peak_counts = {}
    for lead_peaks in all_peaks:
        for peak in lead_peaks:
            # Allow for small variations in peak location between leads
            for offset in range(-5, 6):
                adjusted_peak = peak + offset
                if 0 <= adjusted_peak < ecg_data.shape[0]:
                    peak_counts[adjusted_peak] = peak_counts.get(adjusted_peak, 0) + 1
    
    # Keep peaks that appear in at least 3 leads
    consensus_peaks = sorted([peak for peak, count in peak_counts.items() if count >= 3])
    
    return np.array(consensus_peaks)

def segment_signal(signal, r_peaks, labels, window_size=300):
    """
    Segment the signal into fixed windows around R-peaks.
    """
    pre_buffer = window_size // 2
    post_buffer = window_size - pre_buffer

    segments = []
    valid_labels = []

    for peak in r_peaks:
        segment = np.zeros((window_size, signal.shape[1]))  # Pre-allocate with zeros
        
        # Calculate valid indices
        start_idx = max(0, peak - pre_buffer)
        end_idx = min(signal.shape[0], peak + post_buffer)
        seg_start = pre_buffer - (peak - start_idx)
        seg_end = seg_start + (end_idx - start_idx)
        
        # Copy signal data into zero-padded segment
        segment[seg_start:seg_end] = signal[start_idx:end_idx]
        segments.append(segment)
        valid_labels.append(labels)
    
    return segments, valid_labels

def process_ecg_records(database_path, data_entries, snomed_ct_mapping, peaks_per_signal=10, window_size=300, max_records=None):
    """
    Process ECG records and extract segments around R-peaks.
    """
    X = []
    Y = []
    diagnosis_counts = {}
    processed_records = 0
    skipped_records = 0
    no_relevant_labels = 0
    
    if max_records is not None:
        random.seed(42)
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
    
    total_records = len(data_entries)
    logging.info(f"Starting to process data. Max records: {max_records}, Peaks per signal: {peaks_per_signal}")
    logging.info(f"Number of data entries: {total_records}")

    for i, record in enumerate(data_entries):
        if i % 1000 == 0:
            logging.info(f"Processing record {i}/{total_records}")

        # Construct file paths correctly
        record_path = os.path.join(database_path, record)  # Base path without extension
        mat_file = record_path + '.mat'
        hea_file = record_path  # Don't add .hea, wfdb.rdheader will add it

        if not os.path.exists(mat_file) or not os.path.exists(hea_file + '.hea'):  # Check for .hea existence
            logging.warning(f"Files not found for record {record}")
            skipped_records += 1
            continue

        try:
            # Load ECG data
            mat_data = loadmat(mat_file)
            if 'val' not in mat_data:
                logging.warning(f"'val' key not found in mat file for record {record}")
                skipped_records += 1
                continue
            ecg_data = mat_data['val'].T

            # Read the header file - pass path without .hea extension
            try:
                record_header = wfdb.rdheader(record_path)
            except Exception as e:
                logging.error(f"Error reading header for record {record}: {str(e)}")
                skipped_records += 1
                continue

            # Extract SNOMED-CT codes and map to labels
            snomed_codes = extract_snomed_ct_codes(record_header)
            valid_classes = []
            for code in snomed_codes:
                if code in snomed_ct_mapping:
                    class_name = snomed_ct_mapping[code]
                    valid_classes.append(class_name)
                    diagnosis_counts[class_name] = diagnosis_counts.get(class_name, 0) + 1

            # Count and skip if no valid classes found
            if not valid_classes:
                logging.warning(f"No relevant labels found for record {record}")
                no_relevant_labels += 1
                skipped_records += 1
                continue

            # Remove duplicates
            valid_classes = list(set(valid_classes))

            # Detect R-peaks
            peaks = detect_r_peaks(ecg_data)
            if len(peaks) == 0:
                # If no peaks found, skip this record
                logging.warning(f"No R-peaks detected in record {record}")
                skipped_records += 1
                continue

            # Skip first and last peaks
            if len(peaks) > 2:
                peaks = peaks[1:-1]

            # Limit peaks per signal
            if len(peaks) > peaks_per_signal:
                peaks = peaks[:peaks_per_signal]

            # Segment signals
            segments, labels = segment_signal(ecg_data, peaks, valid_classes, window_size=window_size)
            if not segments:
                logging.warning(f"No valid segments extracted from record {record}")
                skipped_records += 1
                continue

            X.extend(segments)
            Y.extend(labels)
            processed_records += 1

        except Exception as e:
            logging.error(f"Error processing record {record}: {str(e)}")
            skipped_records += 1
            continue

    logging.info(f"Processed {processed_records} records")
    logging.info(f"Skipped {skipped_records} records")
    logging.info(f"Records without relevant labels: {no_relevant_labels}")
    logging.info(f"Diagnosis counts: {diagnosis_counts}")

    return X, Y, diagnosis_counts

def save_data(X, Y, label_names_list, output_dir, peaks_per_signal):
    """
    Save preprocessed data to disk in a compressed format, organized by peaks_per_signal.
    """
    # Create directory structure: csnecg_preprocessed_data/peaks_N/
    specific_dir = os.path.join(output_dir, f'peaks_{peaks_per_signal}')
    os.makedirs(specific_dir, exist_ok=True)
    
    # Save arrays in compressed format
    X = np.array(X, dtype=np.float32)
    np.savez_compressed(os.path.join(specific_dir, 'X.npz'), X)
    np.savez_compressed(os.path.join(specific_dir, 'Y.npz'), Y)
    np.savez_compressed(os.path.join(specific_dir, 'label_names.npz'), label_names_list)
    
    logging.info(f"Saved compressed data to '{specific_dir}'")

def load_data_numpy(data_dir, peaks_per_signal):
    """
    Load preprocessed data from disk.
    """
    specific_dir = os.path.join(data_dir, f'peaks_{peaks_per_signal}')
    
    # Load numpy arrays directly (no context manager needed for .npy files)
    X = np.load(os.path.join(specific_dir, 'X.npy'))
    Y = np.load(os.path.join(specific_dir, 'Y.npy'))
    label_names = np.load(os.path.join(specific_dir, 'label_names.npy'), allow_pickle=True)
    
    return X, Y, label_names

def prepare_data_for_training(X, Y, test_size=0.15, val_size=0.15, batch_size=128):
    """
    Split data into training, validation, and test sets, and prepare TensorFlow datasets.
    """
    # Split data into train, validation, and test sets
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y.sum(axis=1)  # Stratify by label combinations
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train_val, Y_train_val, test_size=val_size_adjusted, random_state=42,
        stratify=Y_train_val.sum(axis=1)  # Stratify by label combinations
    )

    # Calculate class weights
    class_weights = []
    n_samples = Y_train.shape[0]
    n_classes = Y_train.shape[1]
    for i in range(n_classes):
        class_count = np.sum(Y_train[:, i])
        weight = (n_samples / (n_classes * class_count))
        class_weights.append(weight)
    
    # Standardize data
    scaler = StandardScaler()
    num_samples_train, num_timesteps, num_channels = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_channels)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples_train, num_timesteps, num_channels)
    
    num_samples_valid = X_valid.shape[0]
    X_valid_scaled = scaler.transform(X_valid.reshape(-1, num_channels)).reshape(num_samples_valid, num_timesteps, num_channels)
    
    num_samples_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_channels)).reshape(num_samples_test, num_timesteps, num_channels)

    # Create TensorFlow datasets with proper repeat for training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_scaled, Y_valid))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, Y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Calculate steps using np.ceil
    steps_per_epoch = int(np.ceil(len(Y_train) / batch_size))
    validation_steps = int(np.ceil(len(Y_valid) / batch_size))
    test_steps = int(np.ceil(len(Y_test) / batch_size))

    return (train_dataset, valid_dataset, test_dataset, 
            steps_per_epoch, validation_steps, test_steps, 
            Y_test, class_weights)

def ensure_data_available(local_data_dir, drive_data_dir, peaks_per_signal):
    """
    Ensure that the preprocessed data is available in the local directory.
    If not, copy the .npz files from Google Drive and unzip them into .npy files.
    """
    import shutil

    # Check if local directory exists and has .npy files
    specific_dir = os.path.join(local_data_dir, f'peaks_{peaks_per_signal}')
    npy_files = ['X.npy', 'Y.npy', 'label_names.npy']
    
    # Check if all .npy files exist
    if os.path.exists(specific_dir) and all(os.path.exists(os.path.join(specific_dir, f)) for f in npy_files):
        print(f"Data already available in {specific_dir}")
        return

    # Construct paths for source files in Google Drive
    drive_peaks_dir = os.path.join(drive_data_dir, f'peaks_{peaks_per_signal}')
    if not os.path.exists(drive_peaks_dir):
        raise FileNotFoundError(f"Data directory not found at {drive_peaks_dir}")

    # Create local directory structure
    os.makedirs(specific_dir, exist_ok=True)

    # Files to process
    npz_files = ['X.npz', 'Y.npz', 'label_names.npz']

    # Copy and unzip each file
    for npz_file in npz_files:
        source_path = os.path.join(drive_peaks_dir, npz_file)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File {npz_file} not found in {drive_peaks_dir}")
        
        print(f"Processing {npz_file}...")
        
        # Load the npz file with allow_pickle=True for label_names
        if npz_file == 'label_names.npz':
            with np.load(source_path, allow_pickle=True) as data:
                npy_path = os.path.join(specific_dir, npz_file.replace('.npz', '.npy'))
                np.save(npy_path, data['arr_0'], allow_pickle=True)
        else:
            with np.load(source_path) as data:
                npy_path = os.path.join(specific_dir, npz_file.replace('.npz', '.npy'))
                np.save(npy_path, data['arr_0'])
        print(f"Saved to {npy_path}")

    print("Data preparation completed.")

def main():
    # Define paths
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
                 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia',
                 'Atrioventricular reentrant tachycardia'],
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

    # Process the ECG records
    peaks_per_signal = 50  # Adjust as needed
    window_size = 300
    X, Y, diagnosis_counts = process_ecg_records(
        database_path=database_path,
        data_entries=data_entries,
        snomed_ct_mapping=snomed_ct_mapping,
        peaks_per_signal=peaks_per_signal,
        window_size=window_size,
        max_records=None
    )

    # Prepare labels
    unique_classes = set(class_name for sublist in Y for class_name in sublist)
    mlb = MultiLabelBinarizer(classes=sorted(unique_classes))
    Y_binarized = mlb.fit_transform(Y)
    label_names_list = mlb.classes_

    # Save data to disk
    output_dir = 'csnecg_preprocessed_data'
    save_data(X, Y_binarized, label_names_list, output_dir, peaks_per_signal)

if __name__ == '__main__':
    main()