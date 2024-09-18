# Import required libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import and storing/handling data
import wfdb
import numpy as np

# Signal Processing
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Performance Metrics
from sklearn.metrics import (accuracy_score, precision_score,recall_score,confusion_matrix,f1_score)

# Neural Network
from tensorflow import keras
from keras import layers

import os

# Ensure output directory exists
output_dir = 'output_plots_roshancnn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define database path
database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

# Helper Functions

# Create Class 'ECG_reading'. An ECG_reading contains the 4 most important
# values we can extract from the MIT-BIH database: the record of the reading,
# the location of the rPeaks, the labels associated with each heartbeat/peak,
# and the signal itself
class ECG_reading:
    def __init__(self, record, signal, rPeaks, labels):
        self.record = record      # Record Number in MIT-BIH database
        self.signal = signal      # The ECG signal contained in the record
        self.rPeaks = rPeaks      # The label locations (happens to be @ rPeak loc.)
        self.labels = labels      # The labels for each heartbeat

# Process a record and perform signal preprocessing
def processRecord(recordNum):
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

    return (ECG_reading(recordNum, scaledSignal, rPeaks, labels))

# Segment the signal into individual heartbeats
def segmentSignal(record, valid_labels, label2Num):
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

# Print performance statistics
def print_stats(predictions, labels):
    print("Accuracy = {0:.1f}%".format(accuracy_score(labels,
                                                      predictions)*100))

    print("Precision = {0:.1f}%".format(precision_score(labels,
                                                        predictions,
                                                        average = 'macro')*100))

    print("Recall = {0:.1f}%".format(recall_score(labels,
                                                  predictions,
                                                  average = 'macro')*100))

    print("F1 Score = {0:.1f}%".format(f1_score(labels,
                                                predictions,
                                                average = 'macro')*100))

# Plot and save the confusion matrix
def showConfusionMatrix(predictions, labels, filename):
    # Create Confusion Matrix
    cfm_data = confusion_matrix(labels, predictions)

    cf_matrix = sns.heatmap(cfm_data, annot=True, fmt = '.0f', square = True,
                            cmap = 'YlGnBu',  # Use a default colormap
                            linewidths = 0.5, linecolor = 'k', cbar = False)

    # Apply Axis Formatting
    cf_matrix.set_xlabel("Predicted Classification")
    cf_matrix.set_ylabel("Actual Classification")
    cf_matrix.xaxis.set_ticklabels(["N", "V","A","R","L","/"])
    cf_matrix.yaxis.set_ticklabels(["N", "V","A","R","L","/"])

    # Save Confusion Matrix to file
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot waveform by class and save the plot
def plotWaveformByClass(record, classification, filename):
    rec = processRecord(record)

    testLabels = np.array(rec.labels)
    classIndex = np.where(testLabels == classification)

    classIndex = classIndex[0]
    if np.size(classIndex) == 0:
        print("Classification is not present in given ECG Recording")
        return

    # Protects against overflowing bounds
    for Index in classIndex:
        tPeak = rec.rPeaks[Index]
        lowerBound = tPeak - 150
        upperBound = tPeak + 150
        if ((lowerBound < 0) or (upperBound > len(rec.signal))):
            continue
        else:
            break

    waveform = rec.signal[lowerBound:upperBound]
    plt.figure()
    plt.plot(np.arange(len(waveform)), waveform)
    plt.xlabel("Sample")
    plt.ylabel("Normalized Voltage")
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Main function
def main():
    # This Section sets up most of the variables and definitions needed for
    # wrangling and processing the MIT-BIH database
    # --------------------------------------

    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # an Array containing all of the data entries of the MIT-BIH database
    # * 102 and 104 do not have same channel 0 (MLII) as the rest
    # * 102, 104, 107, 217 all have paced beats
    # For more information on the individualities of each data Entry,
    # https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#leads
    dataEntries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                   203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
                   222, 223, 228, 230, 231, 232, 233, 234]
    invalidDataEntries = [102, 104]

    # While there are further labels that are used to classify beats, these are the
    # only ones with enough samples within the MIT-BIH dataset to be able to train
    # a NN without bootstrapping or making up data 'A' has the least data,
    # with only 2452 data points
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']

    # Dictionary to allow for quick conversion between label and number.
    # What it looks like:
    # 'N' = 0,
    # 'V' = 1,
    # 'A' = 2,
    # 'R' = 3,
    # 'L' = 4,
    # '/' = 5
    label2Num = dict(zip(valid_labels,
                         np.arange(len(valid_labels))))

    Num2Label = dict(zip(np.arange(len(valid_labels)),
                         valid_labels))

    # Data Wrangling
    # Process All records and consolidate data into a "single" data structure
    X = []                  # Input Signals, array of 300 length arrays
    Y_cl = []               # Num b/w 0 - 5 associated with particular class
    Z = []                  # The Letterlabel

    # Process All records
    for record in dataEntries:
        rec = processRecord(record)
        tX, tY, tZ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
        Z.extend(tZ)

    # Convert to numpy array for ease of use
    X = np.array(X)
    Y_cl = np.array(Y_cl)
    Z = np.array(Z)

    recLabels, labelCounts = np.unique(Y_cl, return_counts= True)
    label_dict = dict(zip(recLabels, labelCounts))

    print("Class distribution in the dataset:")
    for label_num, count in label_dict.items():
        print(f"Class {Num2Label[label_num]}: {count} samples")
    print("Total samples:", Y_cl.shape[0])

    # Split into train/test data
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, Y_cl,
                                                        test_size = 0.10,
                                                        random_state = 12)

    # Further split the training data into training/validation
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(X_train, y_cl_train,
                                                        test_size = 0.10,
                                                        random_state = 87)

    # Create NN Classifier labels
    def create_nn_labels(y_cl):
        y_nn = []
        for label in y_cl:
            nn_Labels_temp = [0]*len(valid_labels)
            nn_Labels_temp[label] = 1
            y_nn.append(nn_Labels_temp)
        return np.array(y_nn)

    y_nn_train = create_nn_labels(y_cl_train)
    y_nn_valid = create_nn_labels(y_cl_valid)
    y_nn_test = create_nn_labels(y_cl_test)

    # The Neural Network
    # Convolutional Neural Network Definition

    # Clear older models, prevents lag/bloat
    keras.backend.clear_session()

    # specify processed data information
    n_timesteps = X_train.shape[1]
    n_features = 1
    n_outputs = len(valid_labels)

    # Reshape data for CNN input
    X_train_cnn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_valid_cnn = X_valid.reshape((X_valid.shape[0], n_timesteps, n_features))
    X_test_cnn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Construct CNN
    CNN = keras.models.Sequential()
    # Block 1
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          input_shape=(n_timesteps, n_features),
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=3))
    CNN.add(layers.Dropout(0.1))

    # Block 2
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=2))
    CNN.add(layers.Dropout(0.1))

    # Block 3
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=5))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(n_outputs, activation='softmax'))

    # Compile Model
    CNN.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # Use to Visualize Neural Network Topology and Trainable Parameters
    print(CNN.summary())

# Import required libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import and storing/handling data
import wfdb
import numpy as np

# Signal Processing
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Performance Metrics
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix,
                             f1_score)

# Neural Network
from tensorflow import keras
from keras import layers

import os

# Ensure output directory exists
output_dir = 'output_plots_roshancnn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define database path
database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

# Helper Functions

# Create Class 'ECG_reading'. An ECG_reading contains the 4 most important
# values we can extract from the MIT-BIH database: the record of the reading,
# the location of the rPeaks, the labels associated with each heartbeat/peak,
# and the signal itself
class ECG_reading:
    def __init__(self, record, signal, rPeaks, labels):
        self.record = record      # Record Number in MIT-BIH database
        self.signal = signal      # The ECG signal contained in the record
        self.rPeaks = rPeaks      # The label locations (happens to be @ rPeak loc.)
        self.labels = labels      # The labels for each heartbeat

# Process a record and perform signal preprocessing
def processRecord(recordNum):
    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # Grab MLII readings. Easier to view Normal Beats with these readings
    rawSignal = wfdb.rdrecord(record_name = os.path.join(database_path, str(recordNum)),
                              channels = [0]).p_signal[:,0]
    # Also, grab the corresponding annotations (labels and rPeaks)
    signalAnnotations = wfdb.rdann(record_name = os.path.join(database_path, str(recordNum)),
                                   extension = 'atr')

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

    return (ECG_reading(recordNum, scaledSignal, rPeaks, labels))

# Segment the signal into individual heartbeats
def segmentSignal(record, valid_labels, label2Num):
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

# Print performance statistics
def print_stats(predictions, labels):
    print("Accuracy = {0:.1f}%".format(accuracy_score(labels,
                                                      predictions)*100))

    print("Precision = {0:.1f}%".format(precision_score(labels,
                                                        predictions,
                                                        average = 'macro')*100))

    print("Recall = {0:.1f}%".format(recall_score(labels,
                                                  predictions,
                                                  average = 'macro')*100))

    print("F1 Score = {0:.1f}%".format(f1_score(labels,
                                                predictions,
                                                average = 'macro')*100))

# Plot and save the confusion matrix
def showConfusionMatrix(predictions, labels, filename):
    # Create Confusion Matrix
    cfm_data = confusion_matrix(labels, predictions)

    cf_matrix = sns.heatmap(cfm_data, annot=True, fmt = '.0f', square = True,
                            cmap = 'YlGnBu',  # Use a default colormap
                            linewidths = 0.5, linecolor = 'k', cbar = False)

    # Apply Axis Formatting
    cf_matrix.set_xlabel("Predicted Classification")
    cf_matrix.set_ylabel("Actual Classification")
    cf_matrix.xaxis.set_ticklabels(["N", "V","A","R","L","/"])
    cf_matrix.yaxis.set_ticklabels(["N", "V","A","R","L","/"])

    # Save Confusion Matrix to file
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot waveform by class and save the plot
def plotWaveformByClass(record, classification, filename):
    rec = processRecord(record)

    testLabels = np.array(rec.labels)
    classIndex = np.where(testLabels == classification)

    classIndex = classIndex[0]
    if np.size(classIndex) == 0:
        print("Classification is not present in given ECG Recording")
        return

    # Protects against overflowing bounds
    for Index in classIndex:
        tPeak = rec.rPeaks[Index]
        lowerBound = tPeak - 150
        upperBound = tPeak + 150
        if ((lowerBound < 0) or (upperBound > len(rec.signal))):
            continue
        else:
            break

    waveform = rec.signal[lowerBound:upperBound]
    plt.figure()
    plt.plot(np.arange(len(waveform)), waveform)
    plt.xlabel("Sample")
    plt.ylabel("Normalized Voltage")
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Main function
def main():
    # This Section sets up most of the variables and definitions needed for
    # wrangling and processing the MIT-BIH database
    # --------------------------------------

    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # an Array containing all of the data entries of the MIT-BIH database
    # * 102 and 104 do not have same channel 0 (MLII) as the rest
    # * 102, 104, 107, 217 all have paced beats
    # For more information on the individualities of each data Entry,
    # https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#leads
    dataEntries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                   203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
                   222, 223, 228, 230, 231, 232, 233, 234]
    invalidDataEntries = [102, 104]

    # While there are further labels that are used to classify beats, these are the
    # only ones with enough samples within the MIT-BIH dataset to be able to train
    # a NN without bootstrapping or making up data 'A' has the least data,
    # with only 2452 data points
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']

    # Dictionary to allow for quick conversion between label and number.
    # What it looks like:
    # 'N' = 0,
    # 'V' = 1,
    # 'A' = 2,
    # 'R' = 3,
    # 'L' = 4,
    # '/' = 5
    label2Num = dict(zip(valid_labels,
                         np.arange(len(valid_labels))))

    Num2Label = dict(zip(np.arange(len(valid_labels)),
                         valid_labels))

    # Data Wrangling
    # Process All records and consolidate data into a "single" data structure
    X = []                  # Input Signals, array of 300 length arrays
    Y_cl = []               # Num b/w 0 - 5 associated with particular class
    Z = []                  # The Letterlabel

    # Process All records
    for record in dataEntries:
        rec = processRecord(record)
        tX, tY, tZ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
        Z.extend(tZ)

    # Convert to numpy array for ease of use
    X = np.array(X)
    Y_cl = np.array(Y_cl)
    Z = np.array(Z)

    recLabels, labelCounts = np.unique(Y_cl, return_counts= True)
    label_dict = dict(zip(recLabels, labelCounts))

    print("Class distribution in the dataset:")
    for label_num, count in label_dict.items():
        print(f"Class {Num2Label[label_num]}: {count} samples")
    print("Total samples:", Y_cl.shape[0])

    # Split into train/test data
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, Y_cl,
                                                        test_size = 0.10,
                                                        random_state = 12)

    # Further split the training data into training/validation
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(X_train, y_cl_train,
                                                        test_size = 0.10,
                                                        random_state = 87)

    # Create NN Classifier labels
    def create_nn_labels(y_cl):
        y_nn = []
        for label in y_cl:
            nn_Labels_temp = [0]*len(valid_labels)
            nn_Labels_temp[label] = 1
            y_nn.append(nn_Labels_temp)
        return np.array(y_nn)

    y_nn_train = create_nn_labels(y_cl_train)
    y_nn_valid = create_nn_labels(y_cl_valid)
    y_nn_test = create_nn_labels(y_cl_test)

    # The Neural Network
    # Convolutional Neural Network Definition

    # Clear older models, prevents lag/bloat
    keras.backend.clear_session()

    # specify processed data information
    n_timesteps = X_train.shape[1]
    n_features = 1
    n_outputs = len(valid_labels)

    # Reshape data for CNN input
    X_train_cnn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_valid_cnn = X_valid.reshape((X_valid.shape[0], n_timesteps, n_features))
    X_test_cnn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Construct CNN
    CNN = keras.models.Sequential()
    # Block 1
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          input_shape=(n_timesteps, n_features),
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=3))
    CNN.add(layers.Dropout(0.1))

    # Block 2
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=2))
    CNN.add(layers.Dropout(0.1))

    # Block 3
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=5))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(n_outputs, activation='softmax'))

    # Compile Model
    CNN.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # Use to Visualize Neural Network Topology and Trainable Parameters
    print(CNN.summary())

# Import required libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import and storing/handling data
import wfdb
import numpy as np

# Signal Processing
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Performance Metrics
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix,
                             f1_score)

# Neural Network
from tensorflow import keras
from keras import layers

import os

# Set Up pyplot font and resolution settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 125

# Ensure output directory exists
output_dir = 'output_plots_roshancnn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define database path
database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

# Helper Functions

# Create Class 'ECG_reading'. An ECG_reading contains the 4 most important
# values we can extract from the MIT-BIH database: the record of the reading,
# the location of the rPeaks, the labels associated with each heartbeat/peak,
# and the signal itself
class ECG_reading:
    def __init__(self, record, signal, rPeaks, labels):
        self.record = record      # Record Number in MIT-BIH database
        self.signal = signal      # The ECG signal contained in the record
        self.rPeaks = rPeaks      # The label locations (happens to be @ rPeak loc.)
        self.labels = labels      # The labels for each heartbeat

# Process a record and perform signal preprocessing
def processRecord(recordNum):
    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # Grab MLII readings. Easier to view Normal Beats with these readings
    rawSignal = wfdb.rdrecord(record_name = os.path.join(database_path, str(recordNum)),
                              channels = [0]).p_signal[:,0]
    # Also, grab the corresponding annotations (labels and rPeaks)
    signalAnnotations = wfdb.rdann(record_name = os.path.join(database_path, str(recordNum)),
                                   extension = 'atr')

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

    return (ECG_reading(recordNum, scaledSignal, rPeaks, labels))

# Segment the signal into individual heartbeats
def segmentSignal(record, valid_labels, label2Num):
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

# Print performance statistics
def print_stats(predictions, labels):
    print("Accuracy = {0:.1f}%".format(accuracy_score(labels,
                                                      predictions)*100))

    print("Precision = {0:.1f}%".format(precision_score(labels,
                                                        predictions,
                                                        average = 'macro')*100))

    print("Recall = {0:.1f}%".format(recall_score(labels,
                                                  predictions,
                                                  average = 'macro')*100))

    print("F1 Score = {0:.1f}%".format(f1_score(labels,
                                                predictions,
                                                average = 'macro')*100))

# Plot and save the confusion matrix
def showConfusionMatrix(predictions, labels, filename):
    # Create Confusion Matrix
    cfm_data = confusion_matrix(labels, predictions)

    colors = ['#ffffff', '#ffffff']
    cf_matrix = sns.heatmap(cfm_data, annot=True, fmt = '.0f', square = True,
                            cmap = sns.color_palette(colors, as_cmap = True),
                            linewidths = 0.5, linecolor = 'k', cbar = False)

    # Apply Axis Formatting
    cf_matrix.set_xlabel("Predicted Classification")
    cf_matrix.set_ylabel("Actual Classification")
    cf_matrix.xaxis.set_ticklabels(["N", "V","A","R","L","/"])
    cf_matrix.yaxis.set_ticklabels(["N", "V","A","R","L","/"])

    # Save Confusion Matrix to file
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot waveform by class and save the plot
def plotWaveformByClass(record, classification, filename):
    rec = processRecord(record)

    testLabels = np.array(rec.labels)
    classIndex = np.where(testLabels == classification)

    classIndex = classIndex[0]
    if np.size(classIndex) == 0:
        print("Classification is not present in given ECG Recording")
        return

    # Protects against overflowing bounds
    for Index in classIndex:
        tPeak = rec.rPeaks[Index]
        lowerBound = tPeak - 150
        upperBound = tPeak + 150
        if ((lowerBound < 0) or (upperBound > len(rec.signal))):
            continue
        else:
            break

    waveform = rec.signal[lowerBound:upperBound]
    plt.figure()
    plt.plot(np.arange(len(waveform)), waveform, c = '#355E3B')
    plt.xlabel("Sample")
    plt.ylabel("Normalized Voltage")
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Main function
def main():
    # This Section sets up most of the variables and definitions needed for
    # wrangling and processing the MIT-BIH database
    # --------------------------------------

    samplingRate = 360          # Sampling Rate of the MIT-BIH Database

    # an Array containing all of the data entries of the MIT-BIH database
    # * 102 and 104 do not have same channel 0 (MLII) as the rest
    # * 102, 104, 107, 217 all have paced beats
    # For more information on the individualities of each data Entry,
    # https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#leads
    dataEntries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                   203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
                   222, 223, 228, 230, 231, 232, 233, 234]
    invalidDataEntries = [102, 104]

    # While there are further labels that are used to classify beats, these are the
    # only ones with enough samples within the MIT-BIH dataset to be able to train
    # a NN without bootstrapping or making up data 'A' has the least data,
    # with only 2452 data points
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']

    # Dictionary to allow for quick conversion between label and number.
    # What it looks like:
    # 'N' = 0,
    # 'V' = 1,
    # 'A' = 2,
    # 'R' = 3,
    # 'L' = 4,
    # '/' = 5
    label2Num = dict(zip(valid_labels,
                         np.arange(len(valid_labels))))

    Num2Label = dict(zip(np.arange(len(valid_labels)),
                         valid_labels))

    # Data Wrangling
    # Process All records and consolidate data into a "single" data structure
    X = []                  # Input Signals, array of 300 length arrays
    Y_cl = []               # Num b/w 0 - 5 associated with particular class
    Z = []                  # The Letterlabel

    # Process All records
    for record in dataEntries:
        rec = processRecord(record)
        tX, tY, tZ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
        Z.extend(tZ)

    # Convert to numpy array for ease of use
    X = np.array(X)
    Y_cl = np.array(Y_cl)
    Z = np.array(Z)

    recLabels, labelCounts = np.unique(Y_cl, return_counts= True)
    label_dict = dict(zip(recLabels, labelCounts))

    print("Class distribution in the dataset:")
    for label_num, count in label_dict.items():
        print(f"Class {Num2Label[label_num]}: {count} samples")
    print("Total samples:", Y_cl.shape[0])

    # Split into train/test data
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, Y_cl,
                                                        test_size = 0.10,
                                                        random_state = 12)

    # Further split the training data into training/validation
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(X_train, y_cl_train,
                                                        test_size = 0.10,
                                                        random_state = 87)

    # Create NN Classifier labels
    def create_nn_labels(y_cl):
        y_nn = []
        for label in y_cl:
            nn_Labels_temp = [0]*len(valid_labels)
            nn_Labels_temp[label] = 1
            y_nn.append(nn_Labels_temp)
        return np.array(y_nn)

    y_nn_train = create_nn_labels(y_cl_train)
    y_nn_valid = create_nn_labels(y_cl_valid)
    y_nn_test = create_nn_labels(y_cl_test)

    # The Neural Network
    # Convolutional Neural Network Definition

    # Clear older models, prevents lag/bloat
    keras.backend.clear_session()

    # specify processed data information
    n_timesteps = X_train.shape[1]
    n_features = 1
    n_outputs = len(valid_labels)

    # Reshape data for CNN input
    X_train_cnn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_valid_cnn = X_valid.reshape((X_valid.shape[0], n_timesteps, n_features))
    X_test_cnn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Construct CNN
    CNN = keras.models.Sequential()
    # Block 1
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          input_shape=(n_timesteps, n_features),
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=32,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=3))
    CNN.add(layers.Dropout(0.1))

    # Block 2
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=64,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=2))
    CNN.add(layers.Dropout(0.1))

    # Block 3
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.Conv1D(filters=128,
                          kernel_size=3,
                          activation='relu',
                          padding='same'))
    CNN.add(layers.MaxPooling1D(pool_size=5))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(n_outputs, activation='softmax'))

    # Compile Model
    CNN.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # Use to Visualize Neural Network Topology and Trainable Parameters
    print(CNN.summary())

    # Train CNN
    history = CNN.fit(X_train_cnn,
                      y_nn_train,
                      epochs=20,
                      validation_data=(X_valid_cnn, y_nn_valid),
                      batch_size=512,
                      shuffle=True,
                      verbose=1)

    # Evaluate the model on training data
    print("\nTraining Data Performance")
    y_preds_train = CNN.predict(X_train_cnn)
    y_pred_train = np.argmax(y_preds_train, axis=1)
    y_true_train = np.argmax(y_nn_train, axis=1)
    print_stats(y_pred_train, y_true_train)
    showConfusionMatrix(y_pred_train, y_true_train, 'confusion_matrix_cnn_training.png')

    # Evaluate the model on validation data
    print("\nValidation Data Performance")
    y_preds_valid = CNN.predict(X_valid_cnn)
    y_pred_valid = np.argmax(y_preds_valid, axis=1)
    y_true_valid = np.argmax(y_nn_valid, axis=1)
    print_stats(y_pred_valid, y_true_valid)
    showConfusionMatrix(y_pred_valid, y_true_valid, 'confusion_matrix_cnn_validation.png')

    # Evaluate the model on test data
    print("\nTest Data Performance")
    y_preds_test = CNN.predict(X_test_cnn)
    y_pred_test = np.argmax(y_preds_test, axis=1)
    y_true_test = np.argmax(y_nn_test, axis=1)
    print_stats(y_pred_test, y_true_test)
    showConfusionMatrix(y_pred_test, y_true_test, 'confusion_matrix_cnn_test.png')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'], c='#355E3B', marker='.', label='Training Loss')
    plt.plot(history.history['val_loss'], c='#74C315', marker='.', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'], c='#355E3B', marker='.', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], c='#74C315', marker='.', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

    # Plot example waveforms
    plotWaveformByClass(234, 'V', 'waveform_234_V.png')
    plotWaveformByClass(234, 'N', 'waveform_234_N.png')

if __name__ == '__main__':
    main()