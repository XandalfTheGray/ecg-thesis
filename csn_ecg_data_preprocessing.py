# csn_ecg_data_preprocessing.py

import wfdb
import numpy as np
import os

def load_data(database_path, data_entries, valid_labels, label2Num):
    X, Y_cl = [], []
    for record in data_entries:
        record_path = os.path.join(database_path, record)
        try:
            # Read the ECG signal
            record_data = wfdb.rdrecord(record_path)
            signal = record_data.p_signal  # Shape: (n_samples, n_leads)
            
            # Read annotations
            annotation = wfdb.rdann(record_path, 'atr')
            rPeaks = annotation.sample
            labels = annotation.symbol
            
            # Process each R-peak
            for peak, label in zip(rPeaks, labels):
                if label not in valid_labels:
                    continue  # Skip invalid labels
                
                # Define window around R-peak
                pre_buffer = 150  # Number of samples before R-peak
                post_buffer = 150  # Number of samples after R-peak
                
                # Check boundaries
                if peak - pre_buffer < 0 or peak + post_buffer > len(signal):
                    continue  # Skip if window exceeds signal bounds
                
                # Extract heartbeat segment across all leads
                heartbeat = signal[peak - pre_buffer : peak + post_buffer]  # Shape: (300, n_leads)
                
                # Append to dataset
                X.append(heartbeat)
                Y_cl.append(label2Num[label])
                
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            continue  # Skip problematic records
    return np.array(X), np.array(Y_cl)
