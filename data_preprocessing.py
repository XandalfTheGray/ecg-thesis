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
def processRecord(recordNum, database_path):
    """
    Processes a record and performs signal preprocessing.

    Parameters:
    - recordNum: int, the record number in the MIT-BIH database.
    - database_path: str, the path to the MIT-BIH database.

    Returns:
    - ECG_reading object containing the processed signal, rPeaks, and labels.
    """
    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # Grab MLII readings. Easier to view Normal Beats with these readings
    rawSignal = wfdb.rdrecord(record_name = os.path.join(database_path, str(recordNum)), channels = [0]).p_signal[:,0]
    # Also, grab the corresponding annotations (labels and rPeaks)
    signalAnnotations = wfdb.rdann(record_name = os.path.join(database_path, str(recordNum)),extension = 'atr')

    # Grab the rPeaks and the labels from the annotations
    rPeaks = signalAnnotations.sample
    labels = signalAnnotations.symbol

    # Setup a high-pass filter to remove baseline wander
    order = 2                   # Higher order not necessarily needed
    f0 = 0.5                    # Cut-off frequency
    b, a = scipy.signal.butter(N = order, Wn = f0,
                               btype = 'highpass',
                               fs = samplingRate)
    retSignal = scipy.signal.filtfilt(b, a, rawSignal)   # Apply HP Filter

    # Setup and Apply a 60Hz notch filter to remove powerline hum
    # Second order, quality factor of 10
    f0 = 60
    b, a = scipy.signal.iirnotch(w0 = f0, Q = 10, fs = samplingRate)
    retSignal = scipy.signal.filtfilt(b, a, retSignal)    # Apply Notch

    # Normalize Signal
    retSignal = retSignal.reshape(len(retSignal),1)
    scaler = MinMaxScaler()
    scaledSignal = scaler.fit_transform(retSignal)
    scaledSignal = scaledSignal.reshape(len(retSignal))

    return ECG_reading(recordNum, scaledSignal, rPeaks, labels)

# Segment the signal into individual heartbeats
def segmentSignal(record, valid_labels, label2Num):
    """
    Segments the signal into individual heartbeats.

    Parameters:
    - record: ECG_reading object containing the processed signal, rPeaks, and labels.
    - valid_labels: list, the list of valid labels to segment.
    - label2Num: dict, the dictionary mapping labels to numbers.

    Returns:
    - newSignal: list, the segmented signal.
    - cl_Labels: list, the labels for each heartbeat.
    - classes: list, the class labels.
    """
    # First grab rPeaks, labels, and the signal itself from the record
    labels = record.labels
    rPeaks = np.array(record.rPeaks)
    signal = record.signal

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
