# mitbih_train_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
from datetime import datetime
from collections import Counter
from tensorflow.keras import mixed_precision

# Import models and evaluation
from models import build_cnn, build_resnet18_1d, build_resnet34_1d, build_resnet50_1d, build_transformer
from evaluation import print_stats, showConfusionMatrix

# Import data preprocessing functions
from mitbih_data_preprocessing import processRecord, segmentSignal

def main():
    # Setup
    base_output_dir = 'output_plots'
    dataset_name = 'mitbih'
    model_type = 'transformer' # 'cnn', 'resnet18', 'resnet34', 'resnet50', 'transformer'

    # Create a unique directory name with dataset and model
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select Model & Parameters
    if model_type == 'cnn':
        model_params = {
            'l2_reg': 0.0001,
            'filters': (32, 64, 128),
            'kernel_sizes': (3, 3, 3),
            'dropout_rates': (0.1, 0.1, 0.1, 0.1)
        }
    elif model_type == 'transformer':
        model_params = {
            'head_size': 256,
            'num_heads': 4,
            'ff_dim': 4,
            'num_transformer_blocks': 4,
            'mlp_units': [128],
            'mlp_dropout': 0.4,
            'dropout': 0.25,
        }
    else: #elif model_type == 'resnet18':
        model_params = {
            'l2_reg': 0.001,
            'filters': (32, 64, 128),
            'kernel_sizes': (3, 3, 3),
            'dropout_rates': (0.1, 0.1, 0.1, 0.3)
        }

    # Define data entries and labels
    data_entries = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113',
                    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202',
                    '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '221',
                    '222', '223', '228', '230', '231', '232', '233', '234']
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
    database_path = 'mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/'

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
    X, Y_cl = np.array(X), np.array(Y_cl)
    
    print(f"Loaded data shape - X: {np.array(X).shape}, Y_cl: {np.array(Y_cl).shape}")

    # Count samples per class
    class_counts = Counter(Y_cl)
    print("Class distribution:", dict(class_counts))

    # Filter out classes with less than 10 samples
    min_samples_per_class = 10
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
    
    mask = np.isin(Y_cl, valid_classes)
    X = np.array(X)[mask]
    Y_cl = np.array(Y_cl)[mask]

    print(f"Filtered data shape - X: {X.shape}, Y_cl: {Y_cl.shape}")
    print("Filtered class distribution:", dict(Counter(Y_cl)))

    # Remap labels to be consecutive integers starting from 0
    unique_labels = sorted(set(Y_cl))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    Y_cl = np.array([label_map[y] for y in Y_cl])

    print("Remapped class distribution:", dict(Counter(Y_cl)))

    # Update num_classes and Num2Label
    num_classes = len(unique_labels)
    Num2Label = {i: Num2Label[label] for label, i in label_map.items()}

    # Reshape and Scale
    num_samples, num_timesteps, num_channels = X.shape

    # Scale Data per Channel
    X_reshaped = X.reshape(-1, num_channels)  # Shape: (samples * timesteps, channels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, num_timesteps, num_channels)

    # Split the scaled data
    test_size = 0.2
    val_size = 0.25  # 25% of the remaining 80% = 20% of total

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_cl, test_size=test_size, random_state=42, stratify=Y_cl
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )

    print(f"Train set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_valid.shape}, {y_valid.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")

    # One-Hot Encode
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
    elif model_type == 'transformer':
        # Enable mixed precision for transformer only
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        # Cast input data to float16
        X_train = X_train.astype('float16')
        X_valid = X_valid.astype('float16')
        X_test = X_test.astype('float16')
        
        model = build_transformer(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            activation='softmax',
            **model_params
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

    # Train with adjusted parameters for transformer
    if model_type == 'transformer':
        # Use a smaller batch size and more epochs for transformer
        transformer_batch_size = 64  # Adjust this value based on your GPU memory
        transformer_epochs = 50  # Increase epochs to compensate for smaller batch size
        
        history = model.fit(
            X_train, y_nn_train, 
            epochs=transformer_epochs, 
            validation_data=(X_valid, y_nn_valid),
            batch_size=transformer_batch_size, 
            shuffle=True, 
            callbacks=callbacks, 
            class_weight=class_weight_dict
        )
    else:
        # Original training code for other models
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
            y_pred, y_true, f'confusion_matrix_{name.lower()}.png', output_dir, list(Num2Label.values())
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