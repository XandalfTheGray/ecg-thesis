# data_preprocessing.py

import wfdb
import numpy as np
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
import os

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
    
    print(f"Attempting to process record: {record}")
    print(f"Full path: {record_path}")
    
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

        print(f"Successfully processed record: {record}")
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
