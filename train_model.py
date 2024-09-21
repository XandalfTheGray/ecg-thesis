# train_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
import wfdb
import sys
from datetime import datetime

# Import models and evaluation
from models import build_cnn, build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import print_stats, showConfusionMatrix

# Import data preprocessing functions
from mitbih_data_preprocessing import processRecord, segmentSignal
from csn_ecg_data_preprocessing import load_data as load_csn_data

def main():
    # Setup
    base_output_dir = 'output_plots'
    dataset_name = 'mitbih'  # 'mitbih' or 'csn_ecg'
    model_type = 'cnn' # 'cnn', 'resnet18', 'resnet34', 'resnet50'

    # Create a unique directory name with dataset, model, and datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select Model & Parameters
    model_params = {
        'l2_reg': 0.001,                        # Regularization factor
        'filters': (32, 64, 128),                # Reduced number of filters
        'kernel_sizes': (3, 3, 3),              # Consistent kernel sizes
        'dropout_rates': (0.1, 0.1, 0.1, 0.5)    # Lower dropout rates
    }
  
    
    """
    # Good Resnet18 Params
    model_params = {
        'l2_reg': 0.001,
        'filters': (64, 128, 256, 512),
        'kernel_sizes': (3, 3, 3, 3),
        'dropout_rates': (0.3, 0.3, 0.3, 0.6)
    }
    """
    
    # Define data entries and labels
    if dataset_name == 'mitbih':
        data_entries = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113',
                        '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202',
                        '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '221',
                        '222', '223', '228', '230', '231', '232', '233', '234']
        valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
        database_path = 'mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/'
    elif dataset_name == 'csn_ecg':
        wfdb_dir = os.path.join(database_path, 'WFDBRecords')
        data_entries = []
        
        # Traverse the WFDBRecords directory to gather all record names
        for subdir, dirs, files in os.walk(wfdb_dir):
            for file in files:
                if file.endswith('.hea'):
                    # Get the relative path without the file extension
                    record_path = os.path.join(subdir, file)
                    record_name = os.path.splitext(os.path.relpath(record_path, database_path))[0]
                    data_entries.append(record_name)
        
        valid_labels = ['SR', 'AFIB', 'AT', 'PVC', 'RBBB', 'LBBB', 'APB']
        database_path = 'a-large-scale-ecg-database-1.0.0/a-large-scale-ecg-database-1.0.0/'
    else:
        raise ValueError("Unsupported dataset.")

    label2Num = {label: idx for idx, label in enumerate(valid_labels)}
    Num2Label = {idx: label for idx, label in enumerate(valid_labels)}

    # Load and preprocess data
    if dataset_name == 'mitbih':
        X, Y_cl = [], []
        for record in data_entries:
            ecg_reading = processRecord(record, database_path)
            if ecg_reading is not None:
                segments, labels, _ = segmentSignal(ecg_reading, valid_labels, label2Num)
                X.extend(segments)
                Y_cl.extend(labels)
        X, Y_cl = np.array(X), np.array(Y_cl)
    elif dataset_name == 'csn_ecg':
        X, Y_cl = load_csn_data(database_path, data_entries, valid_labels, label2Num)

    if len(X) == 0 or len(Y_cl) == 0:
        print("Error: No data loaded. Check the data loading process.", file=sys.stderr)
        return

    # Display class distribution
    print("Class distribution:", dict(zip(*np.unique(Y_cl, return_counts=True))))

    # Reshape and Scale
    num_samples, num_timesteps, num_channels = X.shape

    # Scale Data per Channel
    X_reshaped = X.reshape(-1, num_channels)  # Shape: (samples * timesteps, channels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, num_timesteps, num_channels)

    # Split the scaled data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, Y_cl, test_size=0.20, random_state=42, stratify=Y_cl
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # One-Hot Encode
    num_classes = len(valid_labels)
    y_nn_train = keras.utils.to_categorical(y_train, num_classes)
    y_nn_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_nn_test = keras.utils.to_categorical(y_test, num_classes)

    # Class Weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Build Model
    if model_type == 'cnn':
        model = build_cnn(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            **model_params
        )
    elif model_type == 'resnet18':
        model = build_resnet18_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )
    else:
        raise ValueError("Invalid model type.")

    # Compile
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_nn_train, 
        epochs=30, 
        validation_data=(X_valid, y_nn_valid),
        batch_size=256, 
        shuffle=True, 
        callbacks=callbacks, 
        class_weight=class_weight_dict
    )

    # Evaluate
    def evaluate_model(dataset, y_true, name):
        y_pred = np.argmax(model.predict(dataset), axis=1)
        print(f"\n{name} Performance")
        print_stats(y_pred, y_true)
        showConfusionMatrix(
            y_pred, y_true, f'confusion_matrix_{name.lower()}.png', output_dir, valid_labels
        )

    evaluate_model(X_train, y_train, 'Training')
    evaluate_model(X_valid, y_valid, 'Validation')
    evaluate_model(X_test, y_test, 'Test')

    # Plot
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

if __name__ == '__main__':
    main()
