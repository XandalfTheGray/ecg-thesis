# mitbih_data_preprocessing.py

import wfdb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

class ECG_reading:
    def __init__(self, record, signal, rPeaks, labels):
        self.record = record      # Record Number in MIT-BIH database
        self.signal = signal      # The ECG signal contained in the record
        self.rPeaks = rPeaks      # The label locations (happens to be @ rPeak loc.)
        self.labels = labels      # The labels for each heartbeat

# Process a record and perform signal preprocessing
def processRecord(record, database_path):
    # Construct the full path to the record
    record_path = os.path.join(database_path, record)
    
    try:
        # Read the record
        record_data = wfdb.rdrecord(record_path)
        
        # Extract the signal data
        signal = record_data.p_signal
        
        # Try to read annotations
        try:
            ann = wfdb.rdann(record_path, 'atr')
            rPeaks = ann.sample
            labels = ann.symbol
        except Exception as e:
            print(f"Error reading annotations for {record}: {str(e)}")
            rPeaks = None
            labels = None
        
        # Create an ECG_reading object
        ecg_reading = ECG_reading(record, signal, rPeaks, labels)

        return ecg_reading
    except FileNotFoundError:
        print(f"File not found: {record_path}")
        return None
    except Exception as e:
        print(f"Error processing record {record}: {str(e)}")
        return None

# Segment the signal into individual heartbeats
def segmentSignal(record, valid_labels, label2Num):
    # First grab rPeaks, labels, and the signal itself from the record
    labels = record.labels
    rPeaks = record.rPeaks
    signal = record.signal

    if labels is None or rPeaks is None:
        print(f"Labels or rPeaks are None for record: {record.record}")
        return [], [], []

    rPeaks = np.array(rPeaks)

    # How many samples to grab before and after the QRS complex.
    preBuffer = 150
    postBuffer = 150

    # arrays to be returned
    newSignal = []
    cl_Labels = []
    classes = []

    # iterate through all rPeaks. If the label is not valid, skip that peak
    for peakNum in range(1,len(rPeaks)):

        if labels[peakNum] not in valid_labels:
            continue

        # Ensure that we do not grab an incomplete QRS complex
        lowerBound = rPeaks[peakNum] - preBuffer
        upperBound = rPeaks[peakNum] + postBuffer
        if ((lowerBound < 0) or (upperBound > len(signal))):
            continue

        # Randomly undersample from all Normal heartbeats
        if labels[peakNum] == 'N':
            if np.random.uniform(0,1) < 0.85:
                continue

        # if it is valid, grab the 150 samples before and 149 samples after peak
        QRS_Complex = signal[lowerBound:upperBound]

        # Fix the corresponding labels to the data
        newSignal.append(QRS_Complex)
        cl_Labels.append(label2Num[labels[peakNum]])

        classes.append(labels[peakNum])

    return newSignal, cl_Labels, classes

def create_nn_labels(y_cl, num_classes):
    """
    Creates one-hot encoded labels for neural networks.

    Parameters:
    - y_cl: list, the class labels.
    - num_classes: int, the number of classes.

    Returns:
    - y_nn: np.array, the one-hot encoded labels.
    """
    y_nn = []
    for label in y_cl:
        nn_Labels_temp = [0]*num_classes
        nn_Labels_temp[label] = 1
        y_nn.append(nn_Labels_temp)
    return np.array(y_nn)

def load_mitbih_data(data_entries, valid_labels, database_path):
    label2Num = {label: idx for idx, label in enumerate(valid_labels)}
    Num2Label = {idx: label for idx, label in enumerate(valid_labels)}
    print(f"Processing {len(data_entries)} records for MITBIH dataset")
    X, Y_cl = [], []
    for record in data_entries:
        ecg_reading = processRecord(record, database_path)
        if ecg_reading is not None:
            segments, labels, _ = segmentSignal(ecg_reading, valid_labels, label2Num)
            X.extend(segments)
            Y_cl.extend(labels)
        else:
            print(f"Warning: No data for record {record}")
    return np.array(X), np.array(Y_cl), Num2Label

def filter_and_remap_classes(X, Y_cl, min_samples_per_class=10):
    class_counts = Counter(Y_cl)
    print("Initial class distribution:", dict(class_counts))

    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
    mask = np.isin(Y_cl, valid_classes)
    X = X[mask]
    Y_cl = Y_cl[mask]

    print(f"Filtered data shape - X: {X.shape}, Y_cl: {Y_cl.shape}")
    print("Filtered class distribution:", dict(Counter(Y_cl)))

    unique_labels = sorted(set(Y_cl))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    Y_cl = np.array([label_map[y] for y in Y_cl])

    print("Remapped class distribution:", dict(Counter(Y_cl)))
    return X, Y_cl, len(unique_labels), label_map

def scale_data(X):
    num_samples, num_timesteps, num_channels = X.shape
    X_reshaped = X.reshape(-1, num_channels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    return X_scaled.reshape(num_samples, num_timesteps, num_channels)

def split_data(X, Y, test_size=0.2, val_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def prepare_mitbih_data(data_entries, valid_labels, database_path, min_samples_per_class=10):
    # Load data
    X, Y_cl, Num2Label = load_mitbih_data(data_entries, valid_labels, database_path)
    
    # Filter and remap classes
    X, Y_cl, num_classes, label_map = filter_and_remap_classes(X, Y_cl, min_samples_per_class)
    
    # Update Num2Label
    Num2Label = {i: Num2Label[label] for label, i in label_map.items()}
    
    # Scale data
    X_scaled = scale_data(X)
    
    # Split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X_scaled, Y_cl)
    
    print(f"Train set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_valid.shape}, {y_valid.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes, Num2Label
