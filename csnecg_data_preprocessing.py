# csnecg_data_preprocessing.py

import os
import random
import numpy as np
from scipy.io import loadmat
import wfdb
import logging
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import h5py
import tensorflow as tf
import collections

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
    
    Parameters:
    - signal: Raw ECG signal
    - fs: Sampling frequency (Hz)
    - lowcut: Lower frequency bound (Hz)
    - highcut: Upper frequency bound (Hz)
    
    Returns:
    - Filtered signal
    """
    from scipy.signal import butter, filtfilt
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg_data, distance=50, fs=500):
    """
    Detect R-peaks using multiple leads with filtering for more robust detection.
    
    Parameters:
    - ecg_data: 12-lead ECG data (timesteps, 12)
    - distance: Minimum distance between peaks
    - fs: Sampling frequency in Hz
    
    Returns:
    - peaks: Array of R-peak locations
    """
    from scipy.signal import find_peaks
    
    # Use leads I, II, and V1-V6 for peak detection
    important_leads = [0, 1, 6, 7, 8, 9, 10, 11]  # Leads I, II, V1-V6
    
    all_peaks = []
    for lead_idx in important_leads:
        # Filter the signal
        filtered_signal = filter_ecg(ecg_data[:, lead_idx], fs=fs)
        
        # Find peaks in filtered signal
        peaks, _ = find_peaks(filtered_signal, 
                              distance=distance,
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
    If no peaks provided, finds first peak in signal.
    """
    if len(r_peaks) == 0:
        # Try to find at least one peak in any lead
        for lead_idx in [0, 1, 6, 7, 8, 9, 10, 11]:  # Important leads
            # Always filter before peak detection
            filtered_signal = filter_ecg(signal[:, lead_idx], fs=500)
            peaks, _ = find_peaks(filtered_signal, 
                                distance=50,
                                height=0.1,      # Match detect_r_peaks parameters
                                prominence=0.2)
            if len(peaks) > 0:
                r_peaks = [peaks[0]]  # Use first peak found
                break
    
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
    
    return np.array(segments), np.array(valid_labels)

def process_and_save_segments(database_path, data_entries, snomed_ct_mapping, peaks_per_signal=10, max_records=None):
    """Process ECG records and save segments incrementally to an HDF5 file."""
    processed_records = 0
    skipped_records = 0
    diagnosis_counts = {}
    
    try:
        logging.info(f"Starting to process data. Max records: {max_records}, Peaks per signal: {peaks_per_signal}")
        
        if max_records is not None:
            random.seed(42)
            data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
        
        total_records = len(data_entries)
        
        # Create HDF5 filename with peaks_per_signal
        hdf5_file_path = f'csnecg_segments_{peaks_per_signal}peaks.hdf5'
        
        # Configure compression
        compression_opts = {
            'compression': 'gzip',  # Use GZIP compression
            'compression_opts': 9,  # Highest compression level (1-9)
            'shuffle': True,        # Enable shuffle filter
            'chunks': True          # Enable automatic chunking
        }
        
        with h5py.File(hdf5_file_path, 'w') as hdf5_file:
            # Create dataset placeholders with max shape and compression
            segments_dataset = hdf5_file.create_dataset(
                'segments',
                shape=(0, 300, 12),
                maxshape=(None, 300, 12),
                dtype=np.float32,
                **compression_opts
            )
            labels_dataset = hdf5_file.create_dataset(
                'labels',
                shape=(0, ),
                maxshape=(None, ),
                dtype=h5py.special_dtype(vlen=np.dtype('int32')),
                **compression_opts
            )
            label_names_set = set()
            
            for i, record in enumerate(data_entries):
                try:
                    if i % 100 == 0:
                        logging.info(f"Processing record {i}/{total_records}")
                    
                    mat_file = os.path.join(database_path, record + '.mat')
                    hea_file = os.path.join(database_path, record + '.hea')
                    
                    if not os.path.exists(mat_file) or not os.path.exists(hea_file):
                        logging.warning(f"Files not found for record {record}")
                        skipped_records += 1
                        continue
                    
                    # Load ECG data
                    mat_data = loadmat(mat_file)
                    if 'val' not in mat_data:
                        logging.warning(f"'val' key not found in mat file for record {record}")
                        skipped_records += 1
                        continue
                    ecg_data = mat_data['val'].T
                    
                    # Read the header file
                    record_header = wfdb.rdheader(os.path.join(database_path, record))
                    
                    # Extract SNOMED codes and map to labels
                    snomed_codes = extract_snomed_ct_codes(record_header)
                    valid_classes = []
                    for code in snomed_codes:
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
                    
                    # Update label names set
                    label_names_set.update(valid_classes)
                    
                    # Detect R-peaks
                    peaks = detect_r_peaks(ecg_data)
                    if len(peaks) == 0:
                        logging.warning(f"No R-peaks detected in record {record}")
                        skipped_records += 1
                        continue
                    
                    # Limit peaks per signal
                    if len(peaks) > peaks_per_signal:
                        np.random.seed(42)  # For reproducibility
                        peaks = np.random.choice(peaks, peaks_per_signal, replace=False)
                        peaks.sort()  # Keep peaks in order
                    
                    # Segment signals
                    segments, _ = segment_signal(ecg_data, peaks, valid_classes)
                    if segments.size == 0:
                        logging.warning(f"No valid segments extracted from record {record}")
                        skipped_records += 1
                        continue
                    
                    # Append segments and labels to HDF5 datasets
                    num_new_segments = segments.shape[0]
                    segments_dataset.resize(segments_dataset.shape[0] + num_new_segments, axis=0)
                    segments_dataset[-num_new_segments:] = segments.astype(np.float32)
                    
                    # Convert labels to integer indices
                    labels_indices = []
                    for _ in range(num_new_segments):
                        indices = [sorted(label_names_set).index(lbl) for lbl in valid_classes]
                        labels_indices.append(np.array(indices, dtype=np.int32))
                    labels_dataset.resize(labels_dataset.shape[0] + num_new_segments, axis=0)
                    labels_dataset[-num_new_segments:] = labels_indices
                    
                    processed_records += 1
                    
                    if max_records and processed_records >= max_records:
                        break
                        
                except Exception as e:
                    logging.error(f"Error processing record {record}: {str(e)}")
                    skipped_records += 1
                    continue
                
                # Periodically save progress every 10,000 records
                if processed_records % 10000 == 0 and processed_records > 0:
                    try:
                        temp_dir = 'csnecg_preprocessed_data_temp'
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_hdf5_path = os.path.join(temp_dir, f'csnecg_segments_{processed_records}.hdf5')
                        with h5py.File(temp_hdf5_path, 'w') as temp_hdf5:
                            temp_hdf5.create_dataset('segments', data=segments.astype(np.float32))
                            temp_hdf5.create_dataset('labels', data=labels_indices)
                        logging.info(f"Saved temporary progress at {processed_records} records")
                    except Exception as e:
                        logging.error(f"Error saving temporary progress: {str(e)}")
        
            # Save label names
            label_names_list = sorted(label_names_set)
            hdf5_file.create_dataset('label_names', data=np.array(label_names_list, dtype='S'))
            # Save mapping of labels to indices
            label_to_index = {label: idx for idx, label in enumerate(label_names_list)}
            hdf5_file.attrs['label_to_index'] = str(label_to_index)
            
            logging.info(f"Processed {processed_records} records")
            logging.info(f"Skipped {skipped_records} records")
            logging.info(f"Diagnosis counts: {diagnosis_counts}")
            logging.info(f"Data saved to {hdf5_file_path}")
            
    except Exception as e:
        logging.error(f"Critical error in process_and_save_segments: {str(e)}")
        raise

def load_preprocessed_data_generator(hdf5_file_path):
    """Generator function to yield samples from HDF5 file."""
    with h5py.File(hdf5_file_path, 'r') as f:
        segments = f['segments']
        labels = f['labels']
        label_names = [name.decode('utf-8') for name in f['label_names'][()]]
        num_classes = len(label_names)
        num_samples = segments.shape[0]
        
        for i in range(num_samples):
            X_sample = segments[i]  # Shape: (300, 12)
            y_indices = labels[i]
            
            # Convert label indices to multilabel format
            y_multilabel = np.zeros(num_classes)
            y_multilabel[y_indices] = 1  # Set the indices to 1
            
            yield X_sample, y_multilabel

def prepare_csnecg_data(
    base_path, batch_size=128, hdf5_file_path='csnecg_segments.hdf5', max_samples=None
):
    """Prepare CSN ECG data for model training using a generator."""
    try:
        logging.info("Preparing data using generator...")
        file_path = os.path.join(base_path, hdf5_file_path)
        
        # Get metadata from HDF5 file
        with h5py.File(file_path, 'r') as f:
            total_size = f['segments'].shape[0]
            label_names = [name.decode('utf-8') for name in f['label_names'][()]]
            num_classes = len(label_names)
        
        # Calculate correct split sizes
        train_size = int(0.7 * total_size)
        valid_size = int(0.15 * total_size)
        test_size = total_size - train_size - valid_size
        
        # Create dataset from generator with prefetch
        dataset = tf.data.Dataset.from_generator(
            lambda: load_preprocessed_data_generator(file_path),
            output_signature=(
                tf.TensorSpec(shape=(300, 12), dtype=tf.float32),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        # Shuffle with a larger buffer
        dataset = dataset.shuffle(buffer_size=min(50000, total_size))
        
        # Split the dataset
        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        valid_dataset = remaining.take(valid_size)
        test_dataset = remaining.skip(valid_size).take(test_size)
        
        # Batch and cache the datasets
        train_dataset = train_dataset.batch(batch_size).cache().repeat()
        valid_dataset = valid_dataset.batch(batch_size).cache().repeat()
        test_dataset = test_dataset.batch(batch_size).cache()
        
        # Create Num2Label mapping
        Num2Label = {idx: label for idx, label in enumerate(label_names)}
        
        logging.info(f"Data prepared: Train={train_size}, Valid={valid_size}, Test={test_size}")
        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            num_classes,
            label_names,
            Num2Label,
        )
    except Exception as e:
        logging.error(f"Error in prepare_csnecg_data: {str(e)}")
        raise

def check_label_distribution(num_peaks):
    file_path = f'csnecg_segments_{num_peaks}peaks.hdf5'
    with h5py.File(file_path, 'r') as f:
        labels = f['labels'][:]
        flattened_labels = [label for sublist in labels for label in sublist]
        label_counts = collections.Counter(flattened_labels)
        print("Label Distribution:", label_counts)

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

    # Define the number of peaks per signal
    peaks_per_signal = 1

    # Process and save segments with peaks_per_signal limit
    process_and_save_segments(
        database_path=database_path,
        data_entries=data_entries,
        snomed_ct_mapping=snomed_ct_mapping,
        peaks_per_signal=peaks_per_signal,
        max_records=None
    )

    # Check label distribution
    check_label_distribution(peaks_per_signal)

if __name__ == '__main__':
    main()