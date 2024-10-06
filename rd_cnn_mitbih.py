# roshan_cnn_mitbih.py

# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
)
from tensorflow import keras
from keras import layers
import itertools
from matplotlib.colors import ListedColormap

# Set Up pyplot font and resolution settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 125

# Ensure output directory exists
output_dir = 'output_plots_roshancnn'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define database path
database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

# Helper Classes and Functions

class ECG_reading:
    """
    Represents an ECG reading with essential attributes.
    """
    def __init__(self, record, signal, rPeaks, labels):
        self.record = record      # Record Number in MIT-BIH database
        self.signal = signal      # The ECG signal contained in the record
        self.rPeaks = rPeaks      # The label locations (happens to be @ rPeak loc.)
        self.labels = labels      # The labels for each heartbeat

def processRecord(recordNum, samplingRate=360):
    """
    Processes a single ECG record:
    - Reads the signal and annotations.
    - Applies high-pass and notch filters.
    - Normalizes the signal.
    
    Returns:
        ECG_reading object.
    """
    # Read raw ECG signal (channel MLII)
    rawSignal = wfdb.rdrecord(
        record_name=os.path.join(database_path, str(recordNum)),
        channels=[0]
    ).p_signal[:, 0]
    
    # Read annotations
    signalAnnotations = wfdb.rdann(
        record_name=os.path.join(database_path, str(recordNum)),
        extension='atr'
    )
    
    # Extract R-peaks and labels
    rPeaks = signalAnnotations.sample
    labels = signalAnnotations.symbol
    
    # High-pass filter to remove baseline wander
    order = 2
    f0 = 0.5
    b, a = scipy.signal.butter(N=order, Wn=f0, btype='highpass', fs=samplingRate)
    filteredSignal = scipy.signal.filtfilt(b, a, rawSignal)
    
    # Notch filter to remove 60Hz powerline noise
    f0 = 60
    b, a = scipy.signal.iirnotch(w0=f0, Q=10, fs=samplingRate)
    filteredSignal = scipy.signal.filtfilt(b, a, filteredSignal)
    
    # Normalize the signal
    scaler = MinMaxScaler()
    scaledSignal = scaler.fit_transform(filteredSignal.reshape(-1, 1)).flatten()
    
    return ECG_reading(recordNum, scaledSignal, rPeaks, labels)

def removeInvalidPeaks(reading, valid_labels):
    """
    Removes peaks with labels not in valid_labels.
    
    Returns:
        ECG_reading object with filtered rPeaks and labels.
    """
    labels = np.array(reading.labels)
    rPeaks = np.array(reading.rPeaks)
    
    mask = np.isin(labels, valid_labels)
    filtered_rPeaks = rPeaks[mask]
    filtered_labels = labels[mask]
    
    return ECG_reading(
        reading.record,
        reading.signal,
        filtered_rPeaks,
        filtered_labels
    )

def segmentSignal(record, valid_labels, label2Num, preBuffer=150, postBuffer=150):
    """
    Segments the ECG signal into individual heartbeats.
    
    Args:
        record: ECG_reading object.
        valid_labels: List of valid labels.
        label2Num: Dictionary mapping labels to numeric values.
        preBuffer: Number of samples before R-peak.
        postBuffer: Number of samples after R-peak.
    
    Returns:
        Tuple of (newSignal, cl_Labels, classes).
    """
    labels = record.labels
    rPeaks = np.array(record.rPeaks)
    signal = record.signal
    
    newSignal = []
    cl_Labels = []
    classes = []
    
    for peakNum in range(len(rPeaks)):
        if labels[peakNum] not in valid_labels:
            continue
        
        lowerBound = rPeaks[peakNum] - preBuffer
        upperBound = rPeaks[peakNum] + postBuffer
        if lowerBound < 0 or upperBound > len(signal):
            continue
        
        # Undersample Normal heartbeats to balance dataset
        if labels[peakNum] == 'N':
            if np.random.uniform(0, 1) < 0.85:
                continue
        
        QRS_Complex = signal[lowerBound:upperBound]
        newSignal.append(QRS_Complex)
        cl_Labels.append(label2Num[labels[peakNum]])
        classes.append(labels[peakNum])
    
    return newSignal, cl_Labels, classes

def create_nn_labels(y_cl, num_classes):
    """
    Converts numeric class labels to one-hot encoded labels.
    
    Args:
        y_cl: Array of numeric labels.
        num_classes: Total number of classes.
    
    Returns:
        One-hot encoded numpy array.
    """
    return np.eye(num_classes)[y_cl]

def print_stats(predictions, labels):
    """
    Prints performance metrics.
    """
    print(f"Accuracy = {accuracy_score(labels, predictions)*100:.1f}%")
    print(f"Precision = {precision_score(labels, predictions, average='macro')*100:.1f}%")
    print(f"Recall = {recall_score(labels, predictions, average='macro')*100:.1f}%")
    print(f"F1 Score = {f1_score(labels, predictions, average='macro')*100:.1f}%")

def showConfusionMatrix(predictions, labels, filename=None):
    """
    Plots and optionally saves the confusion matrix with square cells, larger font, and no title.
    Labels are displayed right-side up for both axes.
    
    Args:
        predictions: Predicted class labels.
        labels: True class labels.
        filename: If provided, saves the plot to the specified file.
    """
    cfm_data = confusion_matrix(labels, predictions)
    class_labels = ["N", "V", "A", "R", "L", "/"]
    
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased figure size for square shape
    cmap = ListedColormap(['white'])
    sns.heatmap(
        cfm_data, 
        annot=True, 
        fmt='d', 
        cmap=cmap, 
        cbar=False,
        linewidths=1, 
        linecolor='black', 
        xticklabels=class_labels,
        yticklabels=class_labels, 
        ax=ax,
        annot_kws={"size": 16}  # Increased font size for cell values
    )
    
    ax.set_ylabel('Actual Classification', fontsize=16)  # Increased font size
    ax.set_xlabel('Predicted Classification', fontsize=16)  # Increased font size
    
    # Remove title
    ax.set_title('')
    
    # Rotate x-axis labels to be right-side up and increase font size
    plt.xticks(rotation=0, fontsize=14)
    
    # Rotate y-axis labels to be right-side up and increase font size
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    if filename:
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plotWaveformByClass(record, classification, valid_labels, filename=None):
    """
    Plots and optionally saves the first waveform of a given class in a record.
    
    Args:
        record: Record number.
        classification: Class label to plot.
        valid_labels: List of valid labels.
        filename: If provided, saves the plot to the specified file.
    """
    rec = processRecord(record)
    rec = removeInvalidPeaks(rec, valid_labels)
    
    class_indices = np.where(rec.labels == classification)[0]
    if len(class_indices) == 0:
        print(f"Classification '{classification}' not present in record {record}.")
        return
    
    for idx in class_indices:
        tPeak = rec.rPeaks[idx]
        lowerBound = tPeak - 150
        upperBound = tPeak + 150
        if lowerBound < 0 or upperBound > len(rec.signal):
            continue
        waveform = rec.signal[lowerBound:upperBound]
        plt.figure()
        plt.plot(np.arange(len(waveform)), waveform, color='#355E3B')
        plt.xlabel("Sample")
        plt.ylabel("Normalized Voltage")
        plt.title(f"Record {record} - Class '{classification}'")
        if filename:
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        else:
            plt.show()
        break
    else:
        print(f"No valid waveform found for classification '{classification}' in record {record}.")

# Main function
def main():
    # Define valid labels
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
    
    # Create label mappings
    label2Num = {label: idx for idx, label in enumerate(valid_labels)}
    Num2Label = {idx: label for idx, label in enumerate(valid_labels)}
    num_classes = len(valid_labels)
    
    # List of data entries to process
    dataEntries = [
        100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
        114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
        203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
        222, 223, 228, 230, 231, 232, 233, 234
    ]
    
    # Initialize data containers
    X = []
    Y_cl = []
    Z = []
    
    # Process all records
    for record in dataEntries:
        rec = processRecord(record)
        rec = removeInvalidPeaks(rec, valid_labels)
        tX, tY, tZ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
        Z.extend(tZ)
    
    # Convert to numpy arrays
    X = np.array(X)
    Y_cl = np.array(Y_cl)
    Z = np.array(Z)
    
    # Display class distribution
    recLabels, labelCounts = np.unique(Y_cl, return_counts=True)
    label_dict = {Num2Label[label]: count for label, count in zip(recLabels, labelCounts)}
    
    print("Class distribution in the dataset:")
    for label, count in label_dict.items():
        print(f"Class {label}: {count} samples")
    print(f"Total samples: {len(Y_cl)}")
    
    # Split into train/test data
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(
        X, Y_cl, test_size=0.10, random_state=12, stratify=Y_cl
    )
    
    # Further split the training data into training/validation
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(
        X_train, y_cl_train, test_size=0.10, random_state=87, stratify=y_cl_train
    )
    
    # One-hot encode labels for CNN
    y_nn_train = create_nn_labels(y_cl_train, num_classes)
    y_nn_valid = create_nn_labels(y_cl_valid, num_classes)
    y_nn_test = create_nn_labels(y_cl_test, num_classes)
    
    # Reshape data for CNN input
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_cnn = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Display input and output sizes
    print("\nInput and Output Sizes:")
    print(f"Training   - Input: {X_train_cnn.shape}, Output: {y_nn_train.shape}")
    print(f"Validation - Input: {X_valid_cnn.shape}, Output: {y_nn_valid.shape}")
    print(f"Test       - Input: {X_test_cnn.shape}, Output: {y_nn_test.shape}")
    
    # Construct CNN
    CNN = keras.models.Sequential([
        # Block 1
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=3),
        layers.Dropout(0.1),
        
        # Block 2
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.1),
        
        # Block 3
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=5),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile Model
    CNN.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("CNN Model Summary:")
    CNN.summary()
    
    # Train CNN
    history = CNN.fit(
        X_train_cnn, y_nn_train,
        epochs=20,
        validation_data=(X_valid_cnn, y_nn_valid),
        batch_size=512,
        shuffle=True,
        verbose=1
    )
    
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
    
    # Plot training & validation loss
    plt.figure()
    plt.plot(history.history['loss'], color='#355E3B', marker='.', label='Training Loss')
    plt.plot(history.history['val_loss'], color='#74C315', marker='.', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot training & validation accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], color='#355E3B', marker='.', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='#74C315', marker='.', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()
    
    # Plot example waveforms
    plotWaveformByClass(234, 'V', valid_labels, 'waveform_234_V.png')
    plotWaveformByClass(234, 'N', valid_labels, 'waveform_234_N.png')

if __name__ == '__main__':
    main()