# ECG Classification using CNN
# Based on Roshan's work

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wfdb
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from tensorflow import keras
from keras import layers, losses
import os

# Define database path
database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

# Define constants and helper classes
SAMPLING_RATE = 360  # Sampling rate of the ECG signals

class ECG_reading:
    def __init__(self, record, signal, r_peaks, labels):
        self.record = record
        self.signal = signal
        self.r_peaks = r_peaks
        self.labels = labels

# Define valid labels and create label-to-number mappings
valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
label2num = dict(zip(valid_labels, np.arange(len(valid_labels))))
num2label = dict(zip(np.arange(len(valid_labels)), valid_labels))

def process_record(record):
    base_path = os.path.join(database_path, str(record))
    signals, fields = wfdb.rdsamp(base_path, channels=[0])
    annotation = wfdb.rdann(base_path, 'atr')
    
    r_peaks = annotation.sample
    labels = annotation.symbol
    
    processed_beats = []
    
    for i in range(len(r_peaks)-1):
        if labels[i] in valid_labels:
            peak = r_peaks[i]
            start = peak - 90
            end = peak + 96
            
            if start > 0 and end < len(signals):
                beat = signals[start:end].flatten()
                processed_beats.append(ECG_reading(record, beat, peak, label2num[labels[i]]))
    
    return processed_beats

def plot_waveform_by_class(record, label):
    signals, fields = wfdb.rdsamp(f'{database_path}/{record}', channels=[0])
    annotation = wfdb.rdann(f'{database_path}/{record}', 'atr')
    
    r_peaks = annotation.sample
    labels = annotation.symbol
    
    for i, peak in enumerate(r_peaks):
        if labels[i] == label:
            start = peak - 90
            end = peak + 96
            
            if start > 0 and end < len(signals):
                beat = signals[start:end].flatten()
                plt.figure(figsize=(10, 4))
                plt.plot(beat)
                plt.title(f"Record {record}, Label {label}")
                plt.show()
                break

def show_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Data Preparation
data_entries = [100, 101, 102, 103, 104, 105]  # Add more record numbers as needed

all_data = []
for record in data_entries:
    processed_record = process_record(record)
    all_data.extend(processed_record)

X = np.array([data.signal for data in all_data])
y = np.array([data.labels for data in all_data])

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and compile model
model = keras.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(187, 1)),
    layers.MaxPooling1D(3),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(3),
    layers.Conv1D(64, 5, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32, verbose=1)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics
print("Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("Precision:", precision_score(y_test, y_pred_classes, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_classes, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_classes, average='weighted'))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Show confusion matrix
show_confusion_matrix(y_pred_classes, y_test)

# Example: Plot waveform for a specific class
plot_waveform_by_class(234, 'N')

# Additional analysis: Check all labels in MIT-BIH database
all_labels = []
for record in data_entries:
    rec = process_record(record)
    all_labels.extend([r.labels for r in rec])
    rec_labels, label_counts = np.unique([r.labels for r in rec], return_counts=True)
    print(record, [num2label[l] for l in rec_labels], label_counts)

all_labels = np.array(all_labels)
rec_labels, label_counts = np.unique(all_labels, return_counts=True)
label_dict = dict(zip([num2label[l] for l in rec_labels], label_counts))

print("\nAll labels in MIT-BIH + Counts:")
print(label_dict)