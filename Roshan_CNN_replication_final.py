import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wfdb
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
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

def plot_waveform_by_class(record, label, save_path=None):
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
                if save_path:
                    plt.savefig(save_path)
                plt.close()
                break

def show_confusion_matrix(y_pred, y_true, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Data Preparation
data_entries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
               222, 223, 228, 230, 231, 232, 233, 234]
invalid_data_entries = [102, 104]

all_data = []
for record in data_entries:
    processed_record = process_record(record)
    all_data.extend(processed_record)

X = np.array([data.signal for data in all_data])
y = np.array([data.labels for data in all_data])

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
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
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32, verbose=1)

# Evaluate model
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Get the unique classes in all sets
unique_classes = np.unique(np.concatenate((y_train, y_val, y_test)))
present_labels = [num2label[i] for i in unique_classes]

# Function to create confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_labels, 
                yticklabels=present_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Create confusion matrices
os.makedirs('output_plots', exist_ok=True)
plot_confusion_matrix(y_train, y_train_pred_classes, 'Training Confusion Matrix', 'output_plots/train_confusion_matrix.png')
plot_confusion_matrix(y_val, y_val_pred_classes, 'Validation Confusion Matrix', 'output_plots/val_confusion_matrix.png')
plot_confusion_matrix(y_test, y_test_pred_classes, 'Test Confusion Matrix', 'output_plots/test_confusion_matrix.png')

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

# Calculate metrics for each set
train_metrics = calculate_metrics(y_train, y_train_pred_classes)
val_metrics = calculate_metrics(y_val, y_val_pred_classes)
test_metrics = calculate_metrics(y_test, y_test_pred_classes)

# Create a table of metrics
print("\nMetrics Table:")
print("Set       | Accuracy | Precision | Recall | F1-Score")
print("----------|----------|-----------|--------|----------")
print(f"Training  |  {train_metrics[0]:.4f}  |  {train_metrics[1]:.4f}   | {train_metrics[2]:.4f} | {train_metrics[3]:.4f}")
print(f"Validation|  {val_metrics[0]:.4f}  |  {val_metrics[1]:.4f}   | {val_metrics[2]:.4f} | {val_metrics[3]:.4f}")
print(f"Test      |  {test_metrics[0]:.4f}  |  {test_metrics[1]:.4f}   | {test_metrics[2]:.4f} | {test_metrics[3]:.4f}")

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

# Save training history plot
plt.savefig('output_plots/training_history.png')
plt.close()

# Example: Plot waveform for a specific class and save it
plot_waveform_by_class(234, 'N', 'output_plots/waveform_example.png')

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

# After preparing X and y
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution in the entire dataset:")
for label, count in zip(unique, counts):
    print(f"Class {num2label[label]}: {count}")

# Check class distribution in test set
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("\nClass distribution in the test set:")
for label, count in zip(unique_test, counts_test):
    print(f"Class {num2label[label]}: {count}")