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
from scipy.signal import find_peaks

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# At the top of the file, add constants
CHUNK_SIZE = 1000  # Consistent chunk size for HDF5 storage
SHUFFLE_BUFFER_SIZE = CHUNK_SIZE  # Reduced from CHUNK_SIZE * 5
BATCH_SIZE = CHUNK_SIZE  # Match batch size to chunk size
IMPORTANT_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]  # Leads I, II, V1-V6
PEAK_PARAMS = {
    'distance': 50,
    'height': 0.1,
    'prominence': 0.2
}
BASE_DATABASE_PATH = 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'

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

def detect_r_peaks(ecg_data, distance=50, fs=500, filtered_signals=None):
    """
    Detect R-peaks using multiple leads with filtering for more robust detection.
    
    Parameters:
    - ecg_data: 12-lead ECG data (timesteps, 12)
    - distance: Minimum distance between peaks
    - fs: Sampling frequency in Hz
    - filtered_signals: Optional pre-filtered signals dictionary
    
    Returns:
    - peaks: Array of R-peak locations
    """
    from scipy.signal import find_peaks
    
    all_peaks = []
    for lead_idx in IMPORTANT_LEADS:
        if filtered_signals and lead_idx in filtered_signals:
            filtered_signal = filtered_signals[lead_idx]
        else:
            filtered_signal = filter_ecg(ecg_data[:, lead_idx], fs=fs)
        
        peaks, _ = find_peaks(filtered_signal, **PEAK_PARAMS)
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
        for lead_idx in IMPORTANT_LEADS:  # Important leads
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
    
    # Pre-allocate memory for common operations
    filtered_signals = {}  # Cache for filtered signals
    
    try:
        logging.info(f"Starting to process data. Max records: {max_records}, Peaks per signal: {peaks_per_signal}")
        
        if max_records is not None:
            random.seed(42)
            data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
        
        total_records = len(data_entries)
        
        # Create HDF5 filename with peaks_per_signal
        hdf5_file_path = f'csnecg_segments_{peaks_per_signal}peaks.hdf5'
        
        # Configure compression for segments (3D data)
        segments_compression_opts = {
            'compression': 'gzip',
            'compression_opts': 9,
            'shuffle': True,
            'chunks': (CHUNK_SIZE, 300, 12)
        }
        
        # Configure compression for labels (1D data)
        labels_compression_opts = {
            'compression': 'gzip',
            'compression_opts': 9,
            'shuffle': True,
            'chunks': (CHUNK_SIZE,)  # 1D chunks for 1D data
        }
        
        with h5py.File(hdf5_file_path, 'w') as hdf5_file:
            segments_dataset = hdf5_file.create_dataset(
                'segments',
                shape=(0, 300, 12),
                maxshape=(None, 300, 12),
                dtype=np.float32,
                **segments_compression_opts
            )
            
            labels_dataset = hdf5_file.create_dataset(
                'labels',
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.special_dtype(vlen=np.dtype('int32')),
                **labels_compression_opts
            )
            
            # Add label_names dataset
            label_names_set = set()
            
            # First pass to collect all unique labels
            for record in data_entries:
                try:
                    record_header = wfdb.rdheader(os.path.join(database_path, record))
                    snomed_codes = extract_snomed_ct_codes(record_header)
                    for code in snomed_codes:
                        if code in snomed_ct_mapping:
                            label_names_set.add(snomed_ct_mapping[code])
                except Exception as e:
                    continue
            
            label_names_list = sorted(list(label_names_set))
            hdf5_file.create_dataset('label_names', data=np.array(label_names_list, dtype='S'))
            
            # Create label to index mapping
            label_to_index = {label: idx for idx, label in enumerate(label_names_list)}
            
            # Process in batches
            batch_segments = []
            batch_labels = []
            
            for i, record in enumerate(data_entries):
                try:
                    if i % 1000 == 0:  # Reduced logging frequency
                        logging.info(f"Processing record {i}/{total_records}")
                    
                    # Load ECG data
                    mat_file = os.path.join(database_path, record + '.mat')
                    hea_file = os.path.join(database_path, record + '.hea')
                    
                    if not os.path.exists(mat_file) or not os.path.exists(hea_file):
                        skipped_records += 1
                        continue
                    
                    # Efficient data loading
                    ecg_data = loadmat(mat_file)['val'].T if 'val' in loadmat(mat_file) else None
                    if ecg_data is None:
                        skipped_records += 1
                        continue
                    
                    # Cache filtered signals for frequently used leads
                    if record not in filtered_signals:
                        filtered_signals[record] = {
                            lead: filter_ecg(ecg_data[:, lead], fs=500)
                            for lead in IMPORTANT_LEADS
                        }
                    
                    # Process peaks and segments in one go
                    peaks = detect_r_peaks(ecg_data, filtered_signals=filtered_signals[record])
                    
                    # Skip first and last peaks, and limit to peaks_per_signal
                    if len(peaks) > 2:  # Need at least 3 peaks to skip first and last
                        peaks = peaks[1:-1]  # Skip first and last peaks
                        if len(peaks) > peaks_per_signal:
                            peaks = peaks[:peaks_per_signal]  # Take first n peaks after skipping
                    elif len(peaks) > 0:
                        # If we have less than 3 peaks, just take the middle one
                        peaks = [peaks[len(peaks)//2]]
                    
                    # Extract labels once
                    record_header = wfdb.rdheader(os.path.join(database_path, record))
                    snomed_codes = extract_snomed_ct_codes(record_header)
                    label_indices = [
                        label_to_index[snomed_ct_mapping[code]]
                        for code in snomed_codes
                        if code in snomed_ct_mapping
                    ]
                    label_indices = list(set(label_indices))  # Remove duplicates
                    
                    # Process segments
                    segments, _ = segment_signal(ecg_data, peaks, label_indices)
                    
                    if segments.size > 0:
                        batch_segments.append(segments)
                        batch_labels.extend([label_indices] * len(segments))
                    
                    # Write batch when full
                    if len(batch_segments) >= BATCH_SIZE:
                        _write_batch_to_hdf5(
                            hdf5_file, batch_segments, batch_labels,
                            segments_dataset, labels_dataset
                        )
                        batch_segments = []
                        batch_labels = []
                    
                    # Clear filtered signals cache less frequently
                    if i % 5000 == 0:  # Increased from 1000
                        filtered_signals.clear()
                    
                except Exception as e:
                    logging.warning(f"Error processing record {record}: {str(e)}")
                    continue
                
            # Write remaining batch
            if batch_segments:
                _write_batch_to_hdf5(
                    hdf5_file, batch_segments, batch_labels,
                    segments_dataset, labels_dataset
                )
    
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise

def _write_batch_to_hdf5(hdf5_file, batch_segments, batch_labels, segments_dataset, labels_dataset):
    """Helper function to write batches to HDF5 file."""
    try:
        # Pre-allocate arrays for better performance
        segments = np.concatenate(batch_segments, axis=0)
        labels = np.array(batch_labels, dtype=object)
        
        # Single resize operation
        current_size = segments_dataset.shape[0]
        new_size = current_size + len(segments)
        
        # Bulk write operations
        segments_dataset.resize(new_size, axis=0)
        labels_dataset.resize(new_size, axis=0)
        
        # Write in one operation
        segments_dataset[current_size:new_size] = segments
        labels_dataset[current_size:new_size] = labels
            
    except Exception as e:
        logging.error(f"Error in _write_batch_to_hdf5: {str(e)}")
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
        )
        
        # First shuffle with small buffer
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            seed=42  # For reproducibility
        )
        
        # Split datasets
        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        valid_dataset = remaining.take(valid_size)
        test_dataset = remaining.skip(valid_size)
        
        # Then apply smaller, per-epoch shuffling to training data only
        train_dataset = train_dataset.shuffle(
            buffer_size=min(SHUFFLE_BUFFER_SIZE, train_size),
            reshuffle_each_iteration=True
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Batch and cache the datasets
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
    database_path = os.path.join(BASE_DATABASE_PATH, BASE_DATABASE_PATH, 'WFDBRecords')
    csv_path = os.path.join(BASE_DATABASE_PATH, BASE_DATABASE_PATH, 'ConditionNames_SNOMED-CT.csv')

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